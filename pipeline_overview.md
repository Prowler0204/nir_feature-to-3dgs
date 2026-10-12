# Multimodal 3DGS Pipeline Overview

This document outlines the complete data flow and architectural pipeline for the Multimodal (RGB + NIR) 3D Gaussian Splatting project. You can use this to explain exactly where the data comes from, how it is processed, and where it goes.

## High-Level Data Flow Architecture

The following diagram illustrates the complete forward and backward pass of the modified 3DGS pipeline.

```mermaid
graph TD
    %% Inputs
    subgraph Data_Inputs [Data Inputs]
        RGB_Img["Raw RGB Images 4K"]
        NIR_Img["Raw NIR Images 1080p"]
        COLMAP["COLMAP SfM"]
    end

    %% Preprocessing
    RGB_Img --> COLMAP
    NIR_Img --> COLMAP
    COLMAP --> Poses["Unified Camera Poses"]
    COLMAP --> Sparse["Sparse Point Cloud"]

    %% Initialization
    Sparse --> Init["Initialize 3D Gaussians"]
    Init --> Gaussians[("3D Gaussians <br> XYZ, RGB, f_nir, etc.")]

    %% Forward Pass
    subgraph Forward_Pass [Forward Pass gsplat backend]
        Gaussians --> Raster_RGB["Render RGB View"]
        Gaussians --> Raster_NIR["Render f_nir View & Alpha Map"]
    end

    Poses --> Raster_RGB
    Poses --> Raster_NIR

    %% Target Generation
    subgraph Target_Gen [Target Generation Frozen]
        NIR_Img --> CNN["Frozen NIR CNN Encoder"]
        CNN --> Target_F["Target 32D Feature Map"]
    end

    %% Loss Computation
    subgraph Loss_Comp [Loss Computation]
        Raster_RGB --> Loss_RGB["RGB Loss: L1 + SSIM"]
        RGB_Img --> Loss_RGB
        
        Raster_NIR --> Loss_NIR["NIR Loss: Masked Cosine Similarity"]
        Target_F --> Loss_NIR
    end

    %% Backward Pass
    Loss_RGB --> Backprop["Backpropagation & Optimizer"]
    Loss_NIR --> Backprop
    
    Backprop -. Updates .-> Gaussians
    Backprop -. absgrad .-> Densify["Densification & Pruning"]
    Densify -. Grows .-> Gaussians
```

---

## Step-by-Step Pipeline Explanation

### 1. Data Ingestion & Preprocessing
*   **Inputs:** Raw RGB images and raw NIR images.
*   **Processing:** Both sets of images are fed into COLMAP (Structure-from-Motion) simultaneously.
*   **Outputs:** COLMAP estimates the 3D position and orientation of every camera (Camera Poses) and generates an initial low-density map of the scene (Sparse Point Cloud).

### 2. Gaussian Initialization
*   **Action:** The Sparse Point Cloud from COLMAP is used to spawn the initial set of 3D Gaussians.
*   **Our Modification:** Standard Gaussians only hold Position, Rotation, Scale, Opacity, and RGB color. We expanded the data structure so that every Gaussian now also holds a **32-dimensional `f_nir` feature vector**.

### 3. The Training Loop (Forward Pass)
During training, the system randomly picks a camera pose (either RGB or NIR).
*   **If it picks an RGB camera:** The `gsplat` rendering engine splats the 3D Gaussians onto a 2D canvas, outputting a Rendered RGB Image.
*   **If it picks an NIR camera:** The `gsplat` engine splats the 3D Gaussians, but instead of colors, it blends their 32D `f_nir` vectors, outputting a **Rendered 32D Feature Map**. It also outputs an `alpha_map` (which tracks where the solid objects are vs. empty space).

### 4. Target Generation (The Ground Truth)
*   If the system picked an NIR camera, we need to know what the Gaussians *should* have rendered.
*   We take the actual, real-world Raw NIR Image for that camera pose and pass it through our **Frozen CNN Encoder**. 
*   **Output:** The CNN outputs a highly structured **Target 32D Feature Map**.

### 5. Loss Computation (Scoring the Model)
*   **RGB Loss:** We compare the Rendered RGB against the Raw RGB using standard $L_1$ and SSIM math.
*   **NIR Loss:** We compare the Rendered 32D Map against the Target 32D Map using **Cosine Similarity**. We explicitly multiply this score by the `alpha_map`. 
    *   *Why?* So the model only gets penalized for getting the solid geometry wrong, preventing it from trying to learn NIR features for the empty sky.

### 6. Backward Pass & Densification
*   **Backpropagation:** The computed loss tells the optimizer how to adjust the XYZ, RGB, and `f_nir` values of every Gaussian to make them better for the next loop.
*   **Densification:** We extract the `.absgrad` (absolute gradient) from the `gsplat` backend. This tells the system where the image is blurry. The system then clones or splits Gaussians in those blurry areas to add more detail to the scene.
