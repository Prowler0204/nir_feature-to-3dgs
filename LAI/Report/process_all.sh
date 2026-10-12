#!/bin/bash
set -e

echo "Starting full auto-alignment and post-processing..."

# 1. Bushes
echo "Processing Bushes..."
conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/auto_align_improved.py \
    --ref_ply /home/intern/LAI/Final_PCD/bushes.ply \
    --src_ply /home/intern/LAI/Final_PCD/bushes_3dgs.ply \
    --output_ply /home/intern/LAI/Final_PCD/bushes_3dgs_auto_align.ply

conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/post_processing \
    --input /home/intern/LAI/Final_PCD/bushes_3dgs_auto_align.ply \
    --output /home/intern/LAI/Final_PCD/bushes_3dgs_auto_align_postprocessed.ply

# 2. Outdoor Small Plant
echo "Processing Outdoor Small Plant..."
conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/auto_align_improved.py \
    --ref_ply /home/intern/LAI/Final_PCD/outdoor_small_plant.ply \
    --src_ply /home/intern/LAI/Final_PCD/outdoor_small_plant_3dgs.ply \
    --output_ply /home/intern/LAI/Final_PCD/outdoor_small_plant_3dgs_auto_align.ply

conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/post_processing \
    --input /home/intern/LAI/Final_PCD/outdoor_small_plant_3dgs_auto_align.ply \
    --output /home/intern/LAI/Final_PCD/outdoor_small_plant_3dgs_auto_align_postprocessed.ply

# 3. Tree 1
echo "Processing Tree 1..."
conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/auto_align_improved.py \
    --ref_ply /home/intern/LAI/Final_PCD/tree1_tls.ply \
    --src_ply /home/intern/LAI/Final_PCD/tree1_rgb_3dgs.ply \
    --output_ply /home/intern/LAI/Final_PCD/tree1_rgb_3dgs_auto_align.ply

conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/post_processing \
    --input /home/intern/LAI/Final_PCD/tree1_rgb_3dgs_auto_align.ply \
    --output /home/intern/LAI/Final_PCD/tree1_rgb_3dgs_auto_align_postprocessed.ply

# 4. Tree 5
echo "Processing Tree 5..."
conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/auto_align_improved.py \
    --ref_ply /home/intern/LAI/Final_PCD/indoor_tree5_wp.ply \
    --src_ply /home/intern/LAI/Final_PCD/indoor_tree5_3dgs_wp.ply \
    --output_ply /home/intern/LAI/Final_PCD/indoor_tree5_3dgs_auto_align.ply

conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/post_processing \
    --input /home/intern/LAI/Final_PCD/indoor_tree5_3dgs_auto_align.ply \
    --output /home/intern/LAI/Final_PCD/indoor_tree5_3dgs_auto_align_postprocessed.ply

echo "Done!"
