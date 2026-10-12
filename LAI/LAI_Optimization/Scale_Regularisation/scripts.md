# Walkthrough - 3DGS to TLS Point Cloud Scaling and Alignment

This walkthrough describes the scaling and alignment implementation, the mathematical validation, and step-by-step instructions on how to run the tools.

## 1. What was accomplished
We created two Python scripts in `/home/intern/LAI/LAI_Optimization/Scale_Regularisation/`:
1. [align_3dgs_to_tls.py](file:///home/intern/LAI/LAI_Optimization/Scale_Regularisation/align_3dgs_to_tls.py): The main registration script. It scales, translates, and rotates a 3DGS PLY file to match a TLS reference PLY. All other Gaussian properties are preserved.
2. [test_alignment.py](file:///home/intern/LAI/LAI_Optimization/Scale_Regularisation/test_alignment.py): A self-contained test script that mathematically validates coordinate transformations, log-scale shifting, and quaternion multiplications on a toy Gaussian.

---

## 2. Step 1: Verification of Mathematical Correctness
Before running on large files, run the unit test to verify that the scaling, translation, log-scale shifts, and quaternion orientations are calculated and transformed with 100% precision.

Run this command in your terminal:
```bash
python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/test_alignment.py
```

### Expected Output:
```text
Created mock PLY: mock_input.ply
Applying transformation to 3DGS PLY: scale=2.000000, translation=[10. -5.  2.]
Applying orientation rotation using quaternion (w, x, y, z): [0.70710678 0.         0.         0.70710678]
Saving aligned PLY with 100% data preservation to: mock_output.ply
Transformation complete! Saved 1 points.

--- Verification Results ---
Position: Expected [ 6. -3.  8.], Actual [ 6. -3.  8.]
Log-scale: Expected 1.1931471805599453, Actual [1.1931472 1.1931472 1.1931472]
Quaternion: Expected [0.70710678 0.         0.         0.70710678], Actual [0.70710678 0.         0.         0.70710678]
Opacity (should be unchanged): Expected 0.8, Actual 0.800000011920929

SUCCESS: All transformations verified successfully! Math is 100% correct.
```

---

## 3. Step 2: Registering Bounding Boxes (Coarse Alignment)
We will first run **Bounding Box Alignment** to scale and translate the 3DGS point cloud to match the real-world scale and position of the TLS data. 

We parse the crop boxes from `tree1_config.txt` and `tree2_config.txt`:

### For Tree 1:
- **TLS Reference Bounding Box**: `xmin=-9.37762`, `xmax=-8.83881`, `ymin=5.24463`, `ymax=5.93851`
- **3DGS Source Bounding Box**: `xmin=-1.48595`, `xmax=1.74739`, `ymin=-1.13743`, `ymax=2.6834`

Run this command:
```bash
python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/align_3dgs_to_tls.py \
  --mode bbox \
  --ref_ply /home/intern/LAI/TLS/Output_LAI/tree1_tls.ply \
  --src_ply /home/intern/LAI/TLS/Output_LAI/tree1_rgb_3dgs.ply \
  --output_ply /home/intern/LAI/TLS/Output_LAI/tree1_rgb_3dgs_bbox_aligned.ply \
  --ref_bbox "-9.37762,-8.83881,5.24463,5.93851" \
  --src_bbox "-1.48595,1.74739,-1.13743,2.6834"
```

### For Tree 2:
- **TLS Reference Bounding Box**: `xmin=1.83349`, `xmax=9.22329`, `ymin=-18.3925`, `ymax=-10.045`
- **3DGS Source Bounding Box**: `xmin=-2.39632`, `xmax=1.64958`, `ymin=-3.39274`, `ymax=0.631864`

Run this command:
```bash
python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/align_3dgs_to_tls.py \
  --mode bbox \
  --ref_ply /home/intern/LAI/TLS/Output_LAI/tree2_tls.ply \
  --src_ply /home/intern/LAI/TLS/Output_LAI/tree2_rgb_3dgs.ply \
  --output_ply /home/intern/LAI/TLS/Output_LAI/tree2_rgb_3dgs_bbox_aligned.ply \
  --ref_bbox "1.83349,9.22329,-18.3925,-10.045" \
  --src_bbox "-2.39632,1.64958,-3.39274,0.631864"
```

---

## 4. Step 3: ICP Fine-Tuning (Strict Refinement)
Once the point clouds are roughly aligned and scaled to the same dimension, we run **ICP refinement** with scale refinement disabled (`--refine_scale` omitted) to compute the correct fine rotation and translation.

> [!NOTE]
> Ensure `open3d` is installed inside your active conda environment. If it is missing, run:
> `pip install open3d`

### For Tree 1:
Run ICP starting from the Bounding Box initial guess:
```bash
python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/align_3dgs_to_tls.py \
  --mode icp \
  --ref_ply /home/intern/LAI/TLS/Output_LAI/tree1_tls.ply \
  --src_ply /home/intern/LAI/TLS/Output_LAI/tree1_rgb_3dgs.ply \
  --output_ply /home/intern/LAI/TLS/Output_LAI/tree1_rgb_3dgs_aligned.ply \
  --ref_bbox "-9.37762,-8.83881,5.24463,5.93851" \
  --src_bbox "-1.48595,1.74739,-1.13743,2.6834" \
  --voxel_size 0.05
```

### For Tree 2:
Run ICP starting from the Bounding Box initial guess:
```bash
python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/align_3dgs_to_tls.py \
  --mode icp \
  --ref_ply /home/intern/LAI/TLS/Output_LAI/tree2_tls.ply \
  --src_ply /home/intern/LAI/TLS/Output_LAI/tree2_rgb_3dgs.ply \
  --output_ply /home/intern/LAI/TLS/Output_LAI/tree2_rgb_3dgs_aligned.ply \
  --ref_bbox "1.83349,9.22329,-18.3925,-10.045" \
  --src_bbox "-2.39632,1.64958,-3.39274,0.631864" \
  --voxel_size 0.05
```

---

## 5. Alternative Step: Umeyama Keypoint Alignment
If you ever extract reliable matching 3D coordinates between the TLS scan and 3DGS coordinates (e.g. from CloudCompare or standard keypoints), run the closed-form SVD method.

For example, with 3 picking point coordinates (X1,Y1,Z1; X2,Y2,Z2; X3,Y3,Z3):
```bash
python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/align_3dgs_to_tls.py \
  --mode umeyama \
  --ref_ply /home/intern/LAI/TLS/Output_LAI/tree1_tls.ply \
  --src_ply /home/intern/LAI/TLS/Output_LAI/tree1_rgb_3dgs.ply \
  --output_ply /home/intern/LAI/TLS/Output_LAI/tree1_rgb_3dgs_umeyama.ply \
  --ref_pts "-9.37,5.24,1.5 ; -8.90,5.85,1.2 ; -9.10,5.50,4.2" \
  --src_pts "-1.48,-1.13,1.0 ; 1.70,2.60,0.8 ; 0.10,0.50,3.5"
```
