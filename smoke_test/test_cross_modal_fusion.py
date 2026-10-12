import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_mock_tensors():
    """
    Generates a mock RGB feature tensor and a corresponding NIR feature tensor
    containing a sharp high-contrast object (a square).
    Introduces a deliberate 1-pixel spatial shift to the NIR tensor to simulate parallax.
    """
    B, C, H, W = 1, 1, 64, 64
    
    # Base background noise (variance)
    torch.manual_seed(42)
    rgb = torch.randn((B, C, H, W)) * 0.05
    nir = torch.randn((B, C, H, W)) * 0.05
    
    # Sharp square in RGB
    rgb[:, :, 20:40, 20:40] += 1.0
    
    # Sharp square in NIR with 1-pixel spatial shift (parallax)
    nir[:, :, 20:40, 21:41] += 0.8
    
    return rgb, nir

def compute_gradients(x):
    """
    Computes spatial gradient magnitude using Sobel filters.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

def compute_ncc(a, b):
    """
    Computes Normalized Cross-Correlation between two tensors.
    """
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    
    a_norm = a_flat - a_flat.mean()
    b_norm = b_flat - b_flat.mean()
    
    numerator = torch.sum(a_norm * b_norm)
    denominator = torch.sqrt(torch.sum(a_norm**2) * torch.sum(b_norm**2))
    
    if denominator < 1e-8:
        return 0.0
    return (numerator / denominator).item()

def create_masks():
    """
    Creates masks for the edge, halo zone, and flat background.
    """
    H, W = 64, 64
    
    # Edge mask (the exact boundaries of the RGB square)
    edge_mask = torch.zeros((1, 1, H, W), dtype=torch.bool)
    edge_mask[:, :, 19:22, 20:40] = True # top
    edge_mask[:, :, 38:41, 20:40] = True # bottom
    edge_mask[:, :, 20:40, 19:22] = True # left
    edge_mask[:, :, 20:40, 38:41] = True # right
    
    # Halo zone (dilated area just outside the immediate edge)
    halo_zone = torch.zeros((1, 1, H, W), dtype=torch.bool)
    halo_zone[:, :, 16:19, 16:44] = True # Top outer ring
    halo_zone[:, :, 41:44, 16:44] = True # Bottom outer ring
    halo_zone[:, :, 19:41, 16:19] = True # Left outer ring
    halo_zone[:, :, 19:41, 41:44] = True # Right outer ring
    
    # Flat background mask (away from any edges)
    bg_mask = torch.zeros((1, 1, H, W), dtype=torch.bool)
    bg_mask[:, :, 2:15, 2:15] = True
    
    return edge_mask, halo_zone, bg_mask

def run_cross_modal_test():
    print("--- Running RGB-NIR Cross-Modal Validation ---")
    rgb, nir = generate_mock_tensors()
    edge_mask, halo_zone, bg_mask = create_masks()
    
    # --- MOCK NETWORK LAYER ---
    grad_rgb = compute_gradients(rgb)
    grad_nir = compute_gradients(nir)
    
    # Simplified scale map (gradient ratio map)
    scale_map = (grad_rgb + 1e-5) / (grad_nir + 1e-5)
    
    # Align the variance of the NIR feature maps
    aligned_nir = nir * scale_map
    
    # Fuse them
    fused = rgb + aligned_nir
    
    # Calculate gradients of the fused output
    fused_grad = compute_gradients(fused)
    
    # --- VALIDATION CHECKS ---
    
    # [Check 1] Halo Artifact Detection
    bg_grad_mean = fused_grad[bg_mask].mean().item()
    halo_grad_mean = fused_grad[halo_zone].mean().item()
    
    print(f"\n[Check 1] Halo Artifact Detection:")
    print(f"          Background Variance (Mean Grad): {bg_grad_mean:.4f}")
    print(f"          Halo Zone (Mean Grad):           {halo_grad_mean:.4f}")
    
    # [Check 2] Spatial Mismatch (Structural Preservation)
    # Compute Cross-Correlation between fused edges and original RGB edges
    ncc_score = compute_ncc(fused_grad[edge_mask], grad_rgb[edge_mask])
    
    print(f"\n[Check 2] Spatial Mismatch (Structural Similarity):")
    print(f"          Fused vs RGB Edge NCC Score:     {ncc_score:.4f}")
    
    # [Check 3] Detail Contribution
    fused_edge_grad = fused_grad[edge_mask].mean().item()
    rgb_edge_grad = grad_rgb[edge_mask].mean().item()
    
    print(f"\n[Check 3] Detail Contribution:")
    print(f"          RGB-Only Edge Gradient:          {rgb_edge_grad:.4f}")
    print(f"          Fused Edge Gradient:             {fused_edge_grad:.4f}")
    
    # --- MATHEMATICAL ASSERTIONS ---
    
    # Fail if halo zone gradient drops below 90% of the normal background variance
    assert halo_grad_mean >= bg_grad_mean * 0.90, \
        f"TEST FAILED: Artificial zero-gradient halo detected! Halo zone gradient ({halo_grad_mean:.4f}) dropped below background baseline ({bg_grad_mean:.4f})."
        
    # Fail if structural similarity drops below strict threshold (0.90) due to the 1-pixel shift
    assert ncc_score >= 0.90, \
        f"TEST FAILED: Spatial mismatch detected! Structural similarity ({ncc_score:.4f}) dropped below strict threshold (0.90) due to 1-pixel NIR shift."
        
    # Fail if the fused edge gradients washed out the high frequencies
    assert fused_edge_grad >= rgb_edge_grad, \
        f"TEST FAILED: Detail contribution washed out! Fused edge gradient ({fused_edge_grad:.4f}) is lower than RGB-only ({rgb_edge_grad:.4f})."

    print("\nAll cross-modal fusion validation checks passed successfully!")

if __name__ == '__main__':
    run_cross_modal_test()
