import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from models.nir_encoder import NIREncoder

def pca_to_rgb(features_1, features_2):
    # features: (C, H, W)
    C, H1, W1 = features_1.shape
    _, H2, W2 = features_2.shape
    
    # Flatten and transpose to (N, C)
    f1_flat = features_1.reshape(C, -1).transpose(1, 0).cpu().numpy()
    f2_flat = features_2.reshape(C, -1).transpose(1, 0).cpu().numpy()
    
    # Fit PCA on the target features (f2)
    pca = PCA(n_components=3)
    pca.fit(f2_flat)
    
    # Transform both
    f1_pca = pca.transform(f1_flat)
    f2_pca = pca.transform(f2_flat)
    
    # Normalize globally using the target's min and max so colormaps match exactly
    f2_min = f2_pca.min(axis=0, keepdims=True)
    f2_max = f2_pca.max(axis=0, keepdims=True)
    
    f1_rgb = (f1_pca - f2_min) / (f2_max - f2_min + 1e-6)
    f2_rgb = (f2_pca - f2_min) / (f2_max - f2_min + 1e-6)
    
    f1_rgb = np.clip(f1_rgb, 0, 1).reshape(H1, W1, 3)
    f2_rgb = np.clip(f2_rgb, 0, 1).reshape(H2, W2, 3)
    
    return f1_rgb, f2_rgb

def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    
    gaussians = GaussianModel(dataset.sh_degree, "adam")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # Instantiate NIR Encoder (Frozen)
    nir_encoder = NIREncoder(d=32).to("cuda")
    nir_encoder.eval()
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Find held-out NIR cameras from the test set
    test_cameras = scene.getTestCameras()
    nir_cameras = [cam for cam in test_cameras if "nir" in cam.image_name.lower()]
            
    if len(nir_cameras) == 0:
        # Fallback to train cameras if no NIR camera is in the test set
        nir_cameras = [cam for cam in scene.getTrainCameras() if "nir" in cam.image_name.lower()]
                
    if len(nir_cameras) == 0:
        print("No NIR camera found!")
        return

    # Select up to 5 distributed cameras
    num_examples = min(5, len(nir_cameras))
    step = max(1, len(nir_cameras) // num_examples)
    selected_cams = [nir_cameras[i * step] for i in range(num_examples)]
    
    for idx, nir_cam in enumerate(selected_cams):
        print(f"Generating example {idx+1}/{num_examples} using camera: {nir_cam.image_name}")
        
        # 1. Ground Truth NIR
        gt_image = nir_cam.original_image.cuda() # (3, H, W)
        gt_nir_vis = gt_image.permute(1, 2, 0).cpu().numpy()
        
        # 2. Target NIR Features
        with torch.no_grad():
            nir_input = gt_image.mean(dim=0, keepdim=True).unsqueeze(0) # (1, 1, H, W)
            target_f_nir = nir_encoder(nir_input).squeeze(0) # (32, H/4, W/4)
            
        # 3. Render RGB and f_nir from 3DGS
        with torch.no_grad():
            render_pkg = render(nir_cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
            rendered_rgb = render_pkg["render"] # (3, H, W)
            rendered_f_nir = render_pkg["render_f_nir"] # (32, H, W)
            
            rendered_rgb_vis = rendered_rgb.permute(1, 2, 0).cpu().numpy()
            rendered_rgb_vis = np.clip(rendered_rgb_vis, 0, 1)

        # PCA projection (Target and Rendered)
        print("Computing PCA for feature visualization...")
        rendered_f_nir_pca, target_f_nir_pca = pca_to_rgb(rendered_f_nir, target_f_nir)
        
        # Plotting 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        axes[0, 0].imshow(rendered_rgb_vis)
        axes[0, 0].set_title(f"Rendered RGB (from 3DGS)", fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(rendered_f_nir_pca)
        axes[0, 1].set_title("Rendered NIR Features (PCA of f_nir)", fontsize=14)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(gt_nir_vis)
        axes[1, 0].set_title("Ground Truth NIR (Camera)", fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(target_f_nir_pca)
        axes[1, 1].set_title("Target NIR Features (PCA of frozen encoder output)", fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(args.model_path, f"prompt5_comparison_example_{idx+1}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
