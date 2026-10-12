import cv2
import numpy as np
import torch
import torch.nn.functional as F

def generate_mock_image():
    """
    Generates a mock 14-bit RAW image (0-16383) with flat regions and sharp edge transitions.
    Injects realistic Poisson shot noise and Gaussian read noise.
    """
    img = np.zeros((200, 200), dtype=np.float32)
    
    # Flat region (shadows)
    img[:, :] = 1500 
    
    # Sharp transition to flat bright region (foliage/leaves)
    img[60:140, 60:140] = 13500 
    
    # Inject Realistic Noise
    np.random.seed(42)
    noisy_img = np.random.poisson(img).astype(np.float32)
    noisy_img += np.random.normal(loc=0.0, scale=40.0, size=(200, 200))
    
    # Constrain to 14-bit bounds
    raw_14bit = np.clip(noisy_img, 0, 16383).astype(np.uint16)
    
    # Create masks for validation
    # Edge mask: the boundary of the bright square
    edge_mask = np.zeros((200, 200), dtype=bool)
    edge_mask[55:145, 55:145] = True
    edge_mask[65:135, 65:135] = False
    
    # Flat region mask: the interior of the bright square
    flat_mask = np.zeros((200, 200), dtype=bool)
    flat_mask[80:120, 80:120] = True
    
    return raw_14bit, edge_mask, flat_mask

def compute_gradients(img_tensor):
    """
    Computes spatial gradient magnitude using PyTorch Sobel filters.
    Expects input shape [B, C, H, W].
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    grad_x = F.conv2d(img_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(img_tensor, sobel_y, padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2)

def test_preprocessing():
    print("--- Running Robust NIR Preprocessing Validation ---")
    raw_img, edge_mask, flat_mask = generate_mock_image()
    
    # --- 1. Baseline: Standard 8-bit Min-Max Normalization ---
    raw_float = raw_img.astype(np.float32)
    raw_min, raw_max = raw_float.min(), raw_float.max()
    
    baseline = (raw_float - raw_min) / (raw_max - raw_min) * 255.0
    
    # --- 2. RFDE Pipeline: Local Intensity Scaling via Bilateral Filter ---
    # Extract low-frequency base layer using Bilateral Filter
    base = cv2.bilateralFilter(raw_float, d=15, sigmaColor=20000, sigmaSpace=15)
    
    # Extract high-frequency detail layer
    detail = raw_float - base
    
    # Soft core detail to suppress noise (std ~123, so >400 is edge) but highly boost true edges
    detail_clean = np.where(np.abs(detail) > 400, detail * 3.0, detail * 0.1)
    
    # Compress the base layer dynamic range to 8-bit (Local Intensity Scaling)
    base_comp = (base - raw_min) / (raw_max - raw_min) * 128.0 + 64.0
    
    # Normalize detail layer to match the new 8-bit scale
    detail_norm = detail_clean / (raw_max - raw_min) * 128.0
    
    # Fuse them back together
    fused_float = base_comp + detail_norm
    
    # --- VALIDATION CHECKS ---
    
    # [Check 1] Highlight Clipping
    num_clipped = np.sum(fused_float > 255.0) + np.sum(fused_float < 0.0)
    total_pixels = fused_float.size
    clip_percent = (num_clipped / total_pixels) * 100.0
    
    # Apply strict 8-bit clipping for the final output
    enhanced = np.clip(fused_float, 0, 255)
    
    print(f"[Check 1] Highlight Clipping:")
    print(f"          {clip_percent:.2f}% of pixels are clipped out-of-bounds.")
    if clip_percent > 15.0:
        print("          WARNING: High percentage of clipping detected!")
        
    # Scale both images to strict [0.0, 1.0] for fair mathematical gradient comparison
    baseline_scaled = baseline / 255.0
    enhanced_scaled = enhanced / 255.0
    
    t_base = torch.from_numpy(baseline_scaled).unsqueeze(0).unsqueeze(0)
    t_enh = torch.from_numpy(enhanced_scaled).unsqueeze(0).unsqueeze(0)
    
    # Compute gradients
    grad_base = compute_gradients(t_base).squeeze().numpy()
    grad_enh = compute_gradients(t_enh).squeeze().numpy()
    
    # [Check 2] Noise Amplification Check
    noise_grad_base = grad_base[flat_mask].mean()
    noise_grad_enh = grad_enh[flat_mask].mean()
    noise_amp_ratio = noise_grad_enh / (noise_grad_base + 1e-6)
    
    print(f"\n[Check 2] Noise Amplification (Flat Regions):")
    print(f"          Baseline Noise Gradient: {noise_grad_base:.4f}")
    print(f"          Enhanced Noise Gradient: {noise_grad_enh:.4f}")
    print(f"          Noise Amplification Ratio: {noise_amp_ratio:.2f}x")
    
    # [Check 3] Gradient Enhancement (Scale-Matched)
    edge_grad_base = grad_base[edge_mask].mean()
    edge_grad_enh = grad_enh[edge_mask].mean()
    edge_boost_ratio = edge_grad_enh / (edge_grad_base + 1e-6)
    
    print(f"\n[Check 3] Gradient Enhancement (Scale-Matched Edges):")
    print(f"          Baseline Edge Gradient:  {edge_grad_base:.4f}")
    print(f"          Enhanced Edge Gradient:  {edge_grad_enh:.4f}")
    print(f"          Edge Boost Ratio:        {edge_boost_ratio:.2f}x")
    
    # Mathematical Assertions
    assert edge_grad_enh > edge_grad_base * 1.1, "Validation Failed: Edge gradients were not sufficiently enhanced!"
    assert edge_boost_ratio > noise_amp_ratio, f"Validation Failed: Noise ({noise_amp_ratio:.2f}x) was amplified more than edges ({edge_boost_ratio:.2f}x)!"
    
    print("\nAll validation checks passed successfully!")

if __name__ == '__main__':
    test_preprocessing()
