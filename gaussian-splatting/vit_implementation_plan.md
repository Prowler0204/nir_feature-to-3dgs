# ViT Implementation & Alignment Plan

This document outlines the corrected plan for implementing ViT-based target generation and addressing the fundamental pose alignment issues in the RGB-NIR 3DGS pipeline.

---

## 1. The Core Issue: Pose Alignment (Must Fix First)
Switching to a ViT encoder will **not** fix the "green blob" artifact. The root cause is that standard SIFT in COLMAP fails across the RGB and NIR spectra, resulting in completely misaligned NIR cameras (looking at empty space, $\alpha=0$).
*   **The Fix:** Before generating any new targets, we must fix the camera registration. 
*   **Method A (Cross-Modal Matching):** Use learned matchers robust to cross-spectral data, such as SuperPoint + LightGlue, or RoMa, integrated into COLMAP (e.g., via `hloc`).
*   **Method B (Separate SfM + Alignment):** Run SfM on the NIR images independently, then compute a 3D similarity transform (using manual correspondences or ICP on sparse clouds) to align the NIR cameras to the RGB camera space.

## 2. Precomputing Targets (The ViT Encoder)
Once poses are aligned, we will precompute the ViT features to eliminate VRAM overhead and speed up training.

**Encoder Upgrades:**
*   **Model:** `dinov2_vits14_reg`. The `reg` (register) variant is crucial because it absorbs artifact tokens, providing cleaner, dense patch features (`x_norm_patchtokens`).
*   **Input Preprocessing:** NIR is single-channel. We will replicate it to 3 channels and apply standard ImageNet mean/std normalization before passing it to DINOv2.
*   **Resolution:** We will preserve the 16:9 aspect ratio of the 1080p NIR images, resizing to multiples of 14 (e.g., $924 \times 518 \rightarrow 66 \times 37$ patches) rather than distorting them into a square.

**Dimensionality Reduction (PCA):**
Instead of a random projection, we will use Principal Component Analysis (PCA).
1.  Extract the 384D patch features for all (or a subset of) NIR images.
2.  Fit a PCA model to project 384D $\rightarrow$ 32D.
3.  Apply this frozen PCA transformation to all features and save them as `.pt` tensors (fp16) to disk.

## 3. Training Loop Modifications
*   **Loss Normalization:** Currently, the masked loss drops trivially when $\alpha$ is low (empty space). We must normalize it: `Loss = sum(alpha * (1 - cosine)) / (sum(alpha) + epsilon)`.
*   **Logging:** We will log the mean $\alpha$ per NIR view. If it drops near 0, the camera is misaligned.
*   **Resolution Matching:** Instead of upsampling the low-res ViT targets (which invents blurry details), we will **downsample the rendered 32D map** to the patch grid resolution (e.g., $66 \times 37$) before computing the loss.

## 4. Evaluation Strategy
*   **Baseline:** Keep the random CNN pipeline as a baseline for an ablation study in the final report.
*   **Metrics:** Without ground truth 3D NIR data, evaluate using PCA visualizations of the rendered 32D maps, multi-view feature consistency, and correlation between RGB and NIR features on well-aligned views.
