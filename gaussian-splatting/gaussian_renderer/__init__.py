#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Set up rasterization configuration
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = width / (2.0 * tanfovx)
    fy = height / (2.0 * tanfovy)
    cx = width / 2.0
    cy = height / 2.0
    
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device="cuda", dtype=torch.float32)

    # gsplat expects row-major viewmat
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    
    viewmats = viewmat.unsqueeze(0)
    Ks = K.unsqueeze(0)

    means3D = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling
    opacities = pc.get_opacity.squeeze(-1) # gsplat expects (N,)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    if override_color is None:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    f_nir = pc.get_f_nir
    
    # Concatenate RGB and f_nir to render them together in one pass
    combined_colors = torch.cat([colors_precomp, f_nir], dim=-1) # (N, 3 + 32 = 35)

    # Rasterize using gsplat. Do NOT pass backgrounds here: rasterize_to_pixels
    # internally asserts backgrounds.shape == (H, W, C) which our per-camera color
    # vector (1, 35) does not satisfy. Instead we composite the RGB background
    # manually below using the alpha map that gsplat returns.
    rendered_combined, alphas, meta = rasterization(
        means3D, quats, scales, opacities, combined_colors, viewmats, Ks, width, height,
        packed=False, absgrad=True
    )
    
    # rendered_combined: (1, H, W, 35);  alphas: (1, H, W, 1)
    rendered_combined = rendered_combined.squeeze(0).permute(2, 0, 1) # (35, H, W)
    alpha_map = alphas.squeeze(0).squeeze(-1)                         # (H, W)

    # Manual RGB background compositing: rendered_over_bg = rgb + (1 - alpha) * bg
    # This replicates exactly what gsplat's background tensor would do internally.
    rendered_image = rendered_combined[:3, ...] + (1.0 - alpha_map.unsqueeze(0)) * bg_color.view(3, 1, 1)
    
    # NIR features: zero background (unobserved regions carry no NIR signal).
    # No compositing needed — the raw rasterised feature map is correct as-is.
    rendered_f_nir = rendered_combined[3:, ...]
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    
    screenspace_points = meta["means2d"].squeeze(0)
    try:
        screenspace_points.retain_grad()
    except:
        pass
        
    # Extract depth if available or use a dummy tensor
    depth_image = torch.zeros_like(rendered_image[0:1])
    
    # meta["radii"] is [C, N, 2] since packed=False. We want max 2D radius per Gaussian as a (N,) float tensor
    radii_1d = meta["radii"].squeeze(0).max(dim=-1).values.float()
    visibility_filter = radii_1d > 0
    
    out = {
        "render": rendered_image,
        "render_f_nir": rendered_f_nir,
        "viewspace_points": screenspace_points,
        "visibility_filter" : visibility_filter,
        "radii": radii_1d,
        "depth" : depth_image,
        "meta": meta, # Pass meta back for access to absgrad
        "alpha_map": alpha_map # Pass alpha back for masked NIR loss
    }
    
    return out
