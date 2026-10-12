#!/bin/bash
# Shell script to automate the calculation of geometry metrics for all datasets.
# Make sure to run 'chmod +x run_metrics.sh' before executing.

# Set the F-score distance threshold (e.g., 0.05 meters = 5 cm)
THRESHOLD=0.05

echo "==============================================="
echo "Starting Geometry Evaluation Metrics Automation"
echo "==============================================="

# 1. Bushes Dataset
echo -e "\n[1/4] Evaluating Bushes Dataset..."
# Replace the paths below with the actual paths to your PLY files
REF_BUSHES="/home/intern/LAI/Final_PCD/bushes.ply"
SRC_BUSHES="/home/intern/LAI/Final_PCD/bushes_3dgs_aligned.ply"

if [ -f "$REF_BUSHES" ] && [ -f "$SRC_BUSHES" ]; then
    python3 compute_metrics.py --ref "$REF_BUSHES" --src "$SRC_BUSHES" --threshold $THRESHOLD
else
    echo "Warning: Files for Bushes dataset not found in the current directory."
fi

# 2. Outdoor Small Plant Dataset
echo -e "\n[2/4] Evaluating Outdoor Small Plant Dataset..."
REF_SMALL="/home/intern/LAI/Final_PCD/outdoor_small_plant.ply"
SRC_SMALL="/home/intern/LAI/Final_PCD/outdoor_small_plant__3dgs_aligned.ply"

if [ -f "$REF_SMALL" ] && [ -f "$SRC_SMALL" ]; then
    python3 compute_metrics.py --ref "$REF_SMALL" --src "$SRC_SMALL" --threshold $THRESHOLD
else
    echo "Warning: Files for Outdoor Small Plant dataset not found."
fi

# 3. Tree 1 Dataset
echo -e "\n[3/4] Evaluating Tree 1 Dataset..."
REF_TREE1="/home/intern/LAI/Final_PCD/tree1_tls.ply"
SRC_TREE1="/home/intern/LAI/Final_PCD/tree1_rgb_3dgs_bbox_aligned.ply"

if [ -f "$REF_TREE1" ] && [ -f "$SRC_TREE1" ]; then
    python3 compute_metrics.py --ref "$REF_TREE1" --src "$SRC_TREE1" --threshold $THRESHOLD
else
    echo "Warning: Files for Tree 1 dataset not found."
fi

# 4. Tree 5 Dataset
echo -e "\n[4/4] Evaluating Tree 5 Dataset..."
REF_TREE5="/home/intern/LAI/Final_PCD/indoor_tree5_wp.ply"
SRC_TREE5="/home/intern/LAI/Final_PCD/indoor_tree5_rgb_3dgs_bbox_aligned.ply"

if [ -f "$REF_TREE5" ] && [ -f "$SRC_TREE5" ]; then
    python3 compute_metrics.py --ref "$REF_TREE5" --src "$SRC_TREE5" --threshold $THRESHOLD
else
    echo "Warning: Files for Tree 5 dataset not found."
fi

echo -e "\n==============================================="
echo "Evaluation Complete."
echo "Please copy the output metrics into the LaTeX paper."
echo "==============================================="
