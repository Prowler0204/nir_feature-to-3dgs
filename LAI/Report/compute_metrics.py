#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D is required. Please install it via 'pip install open3d'")
    sys.exit(1)

def compute_metrics(ref_path, src_path, f_score_threshold):
    """
    Computes Point-to-Point Cloud distance metrics between a reference (TLS) 
    and a source (Aligned 3DGS) point cloud.
    """
    print(f"Loading Reference Point Cloud: {ref_path}")
    ref_pcd = o3d.io.read_point_cloud(ref_path)
    
    print(f"Loading Source Point Cloud: {src_path}")
    src_pcd = o3d.io.read_point_cloud(src_path)

    ref_pts = np.asarray(ref_pcd.points)
    src_pts = np.asarray(src_pcd.points)

    if len(ref_pts) == 0 or len(src_pts) == 0:
        print("Error: One of the point clouds is empty.")
        return

    print(f"\n--- Density Statistics ---")
    print(f"TLS Points: {len(ref_pts)}")
    print(f"3DGS Points: {len(src_pts)}")

    print(f"\n--- Computing Distances ---")
    # Compute distances from Source to Reference (Accuracy)
    dists_src_to_ref = src_pcd.compute_point_cloud_distance(ref_pcd)
    dists_src_to_ref = np.asarray(dists_src_to_ref)
    
    # Compute distances from Reference to Source (Completeness)
    dists_ref_to_src = ref_pcd.compute_point_cloud_distance(src_pcd)
    dists_ref_to_src = np.asarray(dists_ref_to_src)

    # Cloud-to-Cloud RMSE (Usually defined symmetrically or bidirectionally)
    # Using the standard combined RMSE
    mse_src_to_ref = np.mean(dists_src_to_ref ** 2)
    mse_ref_to_src = np.mean(dists_ref_to_src ** 2)
    c2c_rmse = np.sqrt((mse_src_to_ref + mse_ref_to_src) / 2.0)
    
    # Chamfer Distance (L2 or L1, usually L2 squared is used in deep learning, but L1 is often used in point cloud processing)
    # Here we use Mean of L1 Distances for interpretability
    chamfer_l1 = (np.mean(dists_src_to_ref) + np.mean(dists_ref_to_src)) / 2.0

    # F-Score Computation
    # Precision: % of source points within threshold to reference
    precision = np.sum(dists_src_to_ref < f_score_threshold) / len(dists_src_to_ref)
    # Recall: % of reference points within threshold to source
    recall = np.sum(dists_ref_to_src < f_score_threshold) / len(dists_ref_to_src)
    
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    print("\n--- Final Metrics ---")
    print(f"Chamfer Distance (Mean L1): {chamfer_l1:.6f} meters")
    print(f"Cloud-to-Cloud RMSE:      {c2c_rmse:.6f} meters")
    print(f"F-Score (@ {f_score_threshold}m):        {f_score * 100:.2f} %")
    print(f"  -> Precision: {precision * 100:.2f}%")
    print(f"  -> Recall:    {recall * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute C2C RMSE, Chamfer Distance, and F-score between two PLY files.")
    parser.add_argument("--ref", required=True, help="Path to reference TLS PLY file.")
    parser.add_argument("--src", required=True, help="Path to aligned 3DGS PLY file.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Distance threshold for F-score computation (default: 0.05 meters).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ref):
        print(f"Error: {args.ref} does not exist.")
        sys.exit(1)
    if not os.path.exists(args.src):
        print(f"Error: {args.src} does not exist.")
        sys.exit(1)
        
    compute_metrics(args.ref, args.src, args.threshold)
