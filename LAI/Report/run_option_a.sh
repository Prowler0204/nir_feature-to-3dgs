#!/bin/bash
set -e

datasets=(
    "bushes.ply bushes_3dgs.ply bushes"
    "outdoor_small_plant.ply outdoor_small_plant_3dgs.ply outdoor"
    "tree1_tls.ply tree1_rgb_3dgs.ply tree1"
    "indoor_tree5_wp.ply indoor_tree5_3dgs_wp.ply tree5"
)

for ds in "${datasets[@]}"; do
    set -- $ds
    ref="/home/intern/LAI/Final_PCD/$1"
    src="/home/intern/LAI/Final_PCD/$2"
    prefix=$3
    auto="/home/intern/LAI/Final_PCD/${prefix}_auto.ply"
    post="/home/intern/LAI/Final_PCD/${prefix}_auto_post.ply"
    
    echo "--- $prefix ---"
    # Auto align
    conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/Scale_Regularisation/auto_align_improved.py --ref_ply "$ref" --src_ply "$src" --output_ply "$auto" > /dev/null 2>&1
    
    # Post process
    conda run -n gaussian_splatting python /home/intern/LAI/LAI_Optimization/post_processing --input "$auto" --output "$post" > /dev/null 2>&1
    
    # Compute metrics for auto
    echo "Auto aligned metrics:"
    python3 /home/intern/LAI/Report/compute_metrics.py --ref "$ref" --src "$auto" --threshold 0.05 | grep -E "Chamfer|C2C|F-score"
    
    # Compute metrics for post
    echo "Post metrics:"
    python3 /home/intern/LAI/Report/compute_metrics.py --ref "$ref" --src "$post" --threshold 0.05 | grep -E "Chamfer|C2C|F-score"
done
