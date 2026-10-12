#!/usr/bin/env python3
"""
Fully Automated 3DGS to TLS Alignment Tool
Uses Principal Component Analysis (PCA) to automatically discover the correct Scale and Orientation,
resolves the 180-degree flip ambiguity, and uses ICP to perfectly lock the alignment.
"""

import open3d as o3d
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
import copy

def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def apply_transformation_and_save(src_ply_data, output_ply_path, scale, R, t):
    print("\n[Save] Applying automated transformation to 100% of original 3DGS data...")
    vertex_data = src_ply_data.elements[0].data
    
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    pts = np.vstack((x, y, z)).T
    
    # Rotate, Scale, and Translate
    pts_transformed = scale * (pts @ R.T) + t
    
    vertex_data['x'] = pts_transformed[:, 0]
    vertex_data['y'] = pts_transformed[:, 1]
    vertex_data['z'] = pts_transformed[:, 2]
    
    if 'scale_0' in vertex_data.dtype.names:
        log_s = np.log(scale)
        vertex_data['scale_0'] += log_s
        vertex_data['scale_1'] += log_s
        vertex_data['scale_2'] += log_s
        
    if 'rot_0' in vertex_data.dtype.names:
        q_R = rotation_matrix_to_quaternion(R)
        for i in range(len(vertex_data)):
            q_old = np.array([
                vertex_data['rot_0'][i],
                vertex_data['rot_1'][i],
                vertex_data['rot_2'][i],
                vertex_data['rot_3'][i]
            ])
            q_new = quaternion_multiply(q_R, q_old)
            vertex_data['rot_0'][i] = q_new[0]
            vertex_data['rot_1'][i] = q_new[1]
            vertex_data['rot_2'][i] = q_new[2]
            vertex_data['rot_3'][i] = q_new[3]
            
    filtered_element = PlyElement.describe(vertex_data, 'vertex')
    new_plydata = PlyData([filtered_element], text=src_ply_data.text, byte_order=src_ply_data.byte_order)
    
    print(f"[Save] Saved fully preserved aligned PLY to: {output_ply_path}")
    new_plydata.write(output_ply_path)

def extract_main_cluster(pcd, voxel_size):
    print("  -> Downsampling and removing outliers...")
    down = pcd.voxel_down_sample(voxel_size)
    cl, ind = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    clean = down.select_by_index(ind)
    
    print("  -> Running DBSCAN clustering to find main body...")
    labels = np.array(clean.cluster_dbscan(eps=voxel_size*3, min_points=10, print_progress=False))
    if len(labels) == 0 or labels.max() < 0:
        return clean
    
    # Find largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return clean
        
    largest_cluster_idx = unique_labels[np.argmax(counts)]
    main_cluster = clean.select_by_index(np.where(labels == largest_cluster_idx)[0])
    return main_cluster

def get_pca(pcd):
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    cov = np.cov(pts_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort descending
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Ensure right-handed
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]
        
    return centroid, eigenvalues, eigenvectors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Automated 3DGS PCA+ICP Alignment Tool")
    parser.add_argument("--ref_ply", required=True, help="Path to TLS reference .ply")
    parser.add_argument("--src_ply", required=True, help="Path to 3DGS source .ply")
    parser.add_argument("--output_ply", required=True, help="Path to save aligned 3DGS .ply")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for processing")
    args = parser.parse_args()

    print("\n[1/5] Loading point clouds...")
    ref_pcd = o3d.io.read_point_cloud(args.ref_ply)
    src_pcd = o3d.io.read_point_cloud(args.src_ply)

    print("\n[2/5] Extracting robust main clusters to ignore floaters/noise...")
    ref_main = extract_main_cluster(ref_pcd, args.voxel_size)
    src_main = extract_main_cluster(src_pcd, args.voxel_size)

    print("\n[3/5] Computing Principal Component Analysis (PCA)...")
    ref_centroid, ref_evals, ref_V = get_pca(ref_main)
    src_centroid, src_evals, src_V = get_pca(src_main)

    # Compute isotropic scale based on the ratio of principal axis standard deviations
    # We take the mean of the ratios of the square roots of eigenvalues
    scale_factors = np.sqrt(ref_evals / src_evals)
    scale = np.mean(scale_factors)
    print(f"  -> Auto-detected Scale Factor: {scale:.6f}")

    # Generate 4 possible rotation matrices to account for 180-degree eigenvector flips
    flips = [
        np.diag([1, 1, 1]),
        np.diag([-1, -1, 1]),
        np.diag([-1, 1, -1]),
        np.diag([1, -1, -1])
    ]
    
    print("\n[4/5] Testing rotation alignments and locking in with ICP...")
    best_R = None
    best_t = None
    best_fitness = -1

    for F in flips:
        V_src_flipped = src_V @ F
        R_base = ref_V @ V_src_flipped.T
        
        # Initial translation to match centroids after scaling and rotation
        t_base = ref_centroid - scale * (R_base @ src_centroid)
        
        # Create a test point cloud
        src_test = copy.deepcopy(src_main)
        # Apply transformation: s * R * x + t
        pts = np.asarray(src_test.points)
        src_test.points = o3d.utility.Vector3dVector(scale * (pts @ R_base.T) + t_base)
        
        # Run fast ICP to snap it into perfect alignment and evaluate
        reg = o3d.pipelines.registration.registration_icp(
            src_test, ref_main, max_correspondence_distance=args.voxel_size * 5,
            init=np.eye(4), # already aligned, so init is identity
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # If this flip perfectly aligned the shapes, it will have the highest fitness (inlier ratio)
        if reg.fitness > best_fitness:
            best_fitness = reg.fitness
            
            # The final transformation is the base PCA transform refined by the ICP transform
            icp_T = reg.transformation
            icp_R = icp_T[:3, :3]
            icp_t = icp_T[:3, 3]
            
            best_R = icp_R @ R_base
            best_t = scale * (icp_R @ t_base) + icp_t if scale == 1.0 else icp_R @ t_base + icp_t
            
    print(f"  -> Best Alignment Found! ICP Fitness Score: {best_fitness:.4f}")

    print("\n[5/5] Reading original 3DGS attributes via plyfile...")
    src_ply_data = PlyData.read(args.src_ply)
    
    apply_transformation_and_save(src_ply_data, args.output_ply, scale, best_R, best_t)
