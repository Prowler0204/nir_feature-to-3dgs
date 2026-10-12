#!/bin/bash
THRESHOLD=0.05
conda run -n gaussian_splatting python3 /home/intern/LAI/Report/compute_metrics.py --ref /home/intern/LAI/Final_PCD/bushes.ply --src /home/intern/LAI/Final_PCD/bushes_3dgs_aligned_postprocessed.ply --threshold $THRESHOLD > /home/intern/LAI/Report/post_metrics.txt
conda run -n gaussian_splatting python3 /home/intern/LAI/Report/compute_metrics.py --ref /home/intern/LAI/Final_PCD/outdoor_small_plant.ply --src /home/intern/LAI/Final_PCD/outdoor_small_plant__3dgs_aligned_postprocessed.ply --threshold $THRESHOLD >> /home/intern/LAI/Report/post_metrics.txt
conda run -n gaussian_splatting python3 /home/intern/LAI/Report/compute_metrics.py --ref /home/intern/LAI/Final_PCD/tree1_tls.ply --src /home/intern/LAI/Final_PCD/tree1_rgb_3dgs_bbox_aligned_postprocessed.ply --threshold $THRESHOLD >> /home/intern/LAI/Report/post_metrics.txt
conda run -n gaussian_splatting python3 /home/intern/LAI/Report/compute_metrics.py --ref /home/intern/LAI/Final_PCD/indoor_tree5_wp.ply --src /home/intern/LAI/Final_PCD/indoor_tree5_rgb_3dgs_bbox_aligned_postprocessed.ply --threshold $THRESHOLD >> /home/intern/LAI/Report/post_metrics.txt
