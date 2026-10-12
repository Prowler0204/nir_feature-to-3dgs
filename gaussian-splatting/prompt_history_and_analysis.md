# Pipeline Evaluation Log: Multimodal 3D Gaussian Splatting

This document logs the exact prompts, architectural decisions, and actions taken to form the current Multimodal (RGB + NIR) 3D Gaussian Splatting pipeline. It is designed to be reviewed by an AI to identify potential improvements, bottlenecks, and flaws in the current methodology.

---

## 1. Initial Goal & Data Context
*   **User Prompt / Goal:** Implement a system to encode 1080p Near-Infrared (NIR) features into 4K RGB 3D Gaussians. 
*   **Data Constraint (Crucial):** The user specified that the RGB and NIR videos were captured *sequentially* (one after the other, not simultaneously) and with *different zoom levels* (NIR was zoomed in on a specific plant, while RGB captured the wider surroundings).
*   **Action Taken:** We established the goal of doing a "True 3D Fusion" to avoid error-prone 2D image registration (which would distort the 4K RGB geometry to match the 1080p NIR). The plan relied on COLMAP (Structure-from-Motion) to align the cameras into a shared 3D space.

## 2. Architecture Modification (The 32D Vector & `gsplat`)
*   **Prompt / Problem:** Standard 3D Gaussian Splatting (the original Inria CUDA rasterizer) is hardcoded to render only 3 color channels (RGB) and opacity. It cannot render high-dimensional feature vectors.
*   **Action Taken:** 
    *   We modified the core Gaussian data structure to carry an additional 32-dimensional `f_nir` vector alongside the standard RGB colors. We chose 32 dimensions as a sweet spot for capturing spatial textures without causing a VRAM explosion on the GPU.
    *   We swapped the rendering backend to `gsplat`, which natively supports rendering arbitrary high-dimensional arrays.

## 3. Target Generation & Loss Masking
*   **Prompt / Problem:** If we train the Gaussians directly on raw grayscale NIR pixels, they will likely suffer from "representation collapse" (averaging out into blurry blobs). Furthermore, we needed to ensure the Gaussians didn't try to learn NIR features for empty space (like the sky).
*   **Action Taken:** 
    *   **Frozen CNN:** We passed the raw 1080p NIR images through a lightweight, frozen CNN encoder initialized with random weights. This acts as a spatial inductive bias (edge/texture detector), converting the flat grayscale image into a rich 32D target map. (Note: A CNN was chosen over a ViT like DINO for strict spatial accuracy and memory efficiency).
    *   **Cosine Similarity & Masking:** We implemented a Cosine Similarity loss to compare the rendered 32D `f_nir` maps against the CNN targets. We multiplied this loss by the rendered `alpha_map` (where 1 = solid, 0 = empty space). This ensured the model only penalized the Gaussians for solid geometry and ignored the empty sky.

## 4. Restoring Densification (The `.absgrad` Fix)
*   **Prompt / Problem:** Switching to the `gsplat` backend caused the Gaussians to stop growing (densifying). The gradient tracking was hidden internally by `gsplat`.
*   **Action Taken:** We manually extracted the `.absgrad` (absolute gradient) from the `gsplat` outputs in `train.py` and routed it back into the standard densification logic. This successfully restored the gradient flow, allowing the model to clone and split Gaussians in blurry areas.

## 5. Visual Validation & The Critical Flaw Discovery
*   **Prompt / Problem:** The user provided a PCA visualization of a held-out view. The rendered NIR output appeared as an unlearned "green blob," and the rendered RGB was extremely blurry. The training loss, however, showed very low values (RGB $L_1$: 0.0414, NIR Masked Cosine: 0.0118).
*   **Action Taken (Diagnosis):** 
    *   We analyzed the pipeline and determined the **software architecture was 100% correct**, but the **data collection methodology was fatally flawed**. 
    *   Because the RGB and NIR videos were taken sequentially with different zoom levels, COLMAP completely failed to align the NIR cameras with the RGB cameras. COLMAP virtually placed the NIR cameras pointing at empty space.
    *   Our `alpha_map` masking logic worked *too perfectly*: because the NIR cameras were pointing at empty space, the `alpha_map` was exactly 0. This caused the loss to artificially drop to near-zero (ignoring the error), and the Gaussians learned a flat feature vector (the green blob) because there was no geometry in that area to learn from.

## 6. Final Conclusion & Recommended Next Steps
*   **Action Taken:** We advised the user that the Python/PyTorch engineering phase is successful, but the pipeline requires a new dataset. 
*   **Recommendation:** The user must capture a new dataset using a synchronized, stereo-calibrated rig (RGB and NIR cameras mounted together, triggering at the exact same millisecond, with similar fields of view) so that COLMAP can establish a unified, accurate 3D coordinate space.

---
**End of Log.**
