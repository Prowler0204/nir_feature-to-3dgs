#!/usr/bin/env python3
"""
3DGS to TLS Point Cloud Scale Regularization and Alignment Tool
Developed for forest canopy analysis (Leaf Area Index estimation).

This script scales, rotates, and translates a 3D Gaussian Splatting (3DGS) PLY file
to align with a Terrestrial Laser Scanning (TLS) reference point cloud in meters.

To prevent ANY data loss:
- The script uses the `plyfile` library to read/write the original PLY file structure.
- We directly transform positions (x, y, z), scale parameters (scale_0..2, in log-space),
  and rotation parameters (rot_0..3, represented as quaternions).
- All other custom attributes (spherical harmonics, opacity, colors, etc.) are copied exactly.
"""

import os
import sys
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

# ==========================================
# 1. Mathematical Utility Functions
# ==========================================

def umeyama(X, Y):
    """
    Computes the optimal isotropic scale, rotation, and translation (s, R, t)
    to align source points X to reference points Y such that Y ~ s * R * X + t.
    Uses the SVD-based closed-form solution (Umeyama, 1991).
    
    X: (N, 3) numpy array (source points, e.g. 3DGS keypoints)
    Y: (N, 3) numpy array (reference points, e.g. TLS keypoints)
    
    Returns:
        s: float (scale factor)
        R: (3, 3) numpy array (rotation matrix)
        t: (3,) numpy array (translation vector)
    """
    n, m = X.shape
    if n < 3:
        raise ValueError("Umeyama alignment requires at least 3 corresponding points.")
        
    # Subtract means
    mu_x = X.mean(axis=0) #centroid of source
    mu_y = Y.mean(axis=0) #centroid of reference
    X_centered = X - mu_x #zero-mean source
    Y_centered = Y - mu_y #zero-mean reference
    
    # Compute variance of source points
    var_x = np.mean(np.sum(X_centered**2, axis=1))
    
    # Compute covariance matrix
    Sigma = (Y_centered.T @ X_centered) / n
    
    # Singular Value Decomposition
    U, D, Vt = np.linalg.svd(Sigma)
    
    # Handle reflection case
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1
        
    # Compute rotation
    R = U @ S @ Vt
    
    # Compute scale factor
    D_signed = D.copy()
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D_signed[-1] *= -1
    s = np.sum(D_signed) / var_x
    
    # Compute translation
    t = mu_y - s * R @ mu_x
    
    return s, R, t


def rotation_matrix_to_quaternion(R):
    """
    Converts a 3x3 rotation matrix R to a unit quaternion (w, x, y, z).
    Uses Sheppard's method for numerical stability.
    """
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


def multiply_quaternions(q1, q2):
    """
    Multiplies two quaternions q1 and q2.
    q1: shape (4,) representing (w, x, y, z) - e.g. the scene rotation
    q2: shape (N, 4) representing (w, x, y, z) - e.g. individual Gaussian rotations
    
    Returns:
        q_new: shape (N, 4) representing the rotated quaternions.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    q_new = np.stack([w, x, y, z], axis=-1)
    
    # Normalize to avoid numerical drift
    norms = np.linalg.norm(q_new, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return q_new / norms


# ==========================================
# 2. Main Alignment logic
# ==========================================

def get_bbox_transform(ref_bbox, src_bbox):
    """
    Calculates scale and translation from bounding boxes.
    ref_bbox, src_bbox: lists of [xmin, xmax, ymin, ymax, zmin, zmax]
    
    If z bounds are not specified (e.g. only 4 elements: [xmin, xmax, ymin, ymax]),
    the scale is computed from X and Y, and Z scale is assumed to be their average.
    """
    if len(ref_bbox) == 4:
        ref_x_min, ref_x_max, ref_y_min, ref_y_max = ref_bbox
        ref_z_min, ref_z_max = 0.0, 0.0
        use_3d_z = False
    else:
        ref_x_min, ref_x_max, ref_y_min, ref_y_max, ref_z_min, ref_z_max = ref_bbox
        use_3d_z = True
        
    if len(src_bbox) == 4:
        src_x_min, src_x_max, src_y_min, src_y_max = src_bbox
        src_z_min, src_z_max = 0.0, 0.0
    else:
        src_x_min, src_x_max, src_y_min, src_y_max, src_z_min, src_z_max = src_bbox
        
    # Compute spans
    ref_dx = ref_x_max - ref_x_min
    ref_dy = ref_y_max - ref_y_min
    src_dx = src_x_max - src_x_min
    src_dy = src_y_max - src_y_min
    
    scale_x = ref_dx / src_dx
    scale_y = ref_dy / src_dy
    
    if use_3d_z:
        ref_dz = ref_z_max - ref_z_min
        src_dz = src_z_max - src_z_min
        scale_z = ref_dz / src_dz
        s = (scale_x + scale_y + scale_z) / 3.0
    else:
        s = (scale_x + scale_y) / 2.0
        
    # Calculate centers
    ref_center = np.array([
        (ref_x_min + ref_x_max) / 2.0,
        (ref_y_min + ref_y_max) / 2.0,
        (ref_z_min + ref_z_max) / 2.0 if use_3d_z else 0.0
    ])
    src_center = np.array([
        (src_x_min + src_x_max) / 2.0,
        (src_y_min + src_y_max) / 2.0,
        (src_z_min + src_z_max) / 2.0 if use_3d_z else 0.0
    ])
    
    # Scale translation
    t = ref_center - s * src_center
    
    # Rotation is Identity for bounding box mode
    R = np.eye(3)
    
    if not use_3d_z:
        # Keep Z translation zero or center it around 0
        t[2] = 0.0
        
    return s, R, t


def compute_bbox_from_pts(pts):
    """
    Computes the 3D bounding box from an array of points.
    Returns: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    return [
        float(np.min(pts[:, 0])), float(np.max(pts[:, 0])),
        float(np.min(pts[:, 1])), float(np.max(pts[:, 1])),
        float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))
    ]


def extract_main_cluster(pcd, voxel_size):
    import open3d as o3d
    # Downsample first to drastically save memory!
    print("  [Progress] Downsampling for robust clustering")
    down_pcd = pcd.voxel_down_sample(voxel_size)
    
    # 1. Remove statistical outliers (noise)
    print("  [Progress] Removing statistical outliers")
    cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_pcd = down_pcd.select_by_index(ind)
    
    # 2. DBSCAN clustering to find the main connected component (the plant)
    print("  [Progress] Running DBSCAN clustering")
    eps = voxel_size * 5
    labels = np.array(inlier_pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=True))
    
    if len(labels) == 0 or labels.max() < 0:
        return inlier_pcd # fallback
        
    counts = np.bincount(labels[labels >= 0])
    largest_cluster_idx = counts.argmax()
    
    largest_cluster_indices = np.where(labels == largest_cluster_idx)[0]
    largest_cluster_pcd = inlier_pcd.select_by_index(largest_cluster_indices)
    
    return largest_cluster_pcd

def get_pcd_bbox(pcd):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return [0, 0, 0, 0, 0, 0]
    return [
        float(pts[:,0].min()), float(pts[:,0].max()),
        float(pts[:,1].min()), float(pts[:,1].max()),
        float(pts[:,2].min()), float(pts[:,2].max())
    ]


def pick_points(pcd, title):
    import open3d as o3d
    import sys
    print(f"\n Point Picking: {title}")
    print("1. Rotate the point cloud using left-click & drag.")
    print("2. Hold [Shift] and left-click to pick a point.")
    print("3. Pick EXACTLY 3 corresponding points (e.g., trunk base, specific branch tip).")
    print("4. Close the window when done.")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    vis.run()  # Blocks until window is closed
    vis.destroy_window()
    
    indices = vis.get_picked_points()
    if len(indices) < 3:
        print(f"Error: You only picked {len(indices)} points. You must pick at least 3 points for Umeyama alignment!")
        sys.exit(1)
        
    pts = np.asarray(pcd.points)[indices]
    print(f"Picked {len(pts)} points successfully.")
    return pts


def auto_align_z_sweep(ref_pcd, src_pcd, voxel_size):
    import open3d as o3d
    print("  [Progress] Running Auto Z-Rotation Sweep (36 angles)...")
    
    best_fitness = -1.0
    best_T = np.eye(4)
    
    # Downsample even more for the sweep to make it blazing fast
    ref_sweep = ref_pcd.voxel_down_sample(voxel_size * 2)
    src_sweep = src_pcd.voxel_down_sample(voxel_size * 2)
    
    # Use a strict distance so the fitness score is actually meaningful!
    # If the distance is too large, every angle will get a fitness of 1.0.
    max_corr_dist = voxel_size * 1.5 
    
    for angle_deg in range(0, 360, 10):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        T_init = np.eye(4)
        T_init[0, 0] = cos_a
        T_init[0, 1] = -sin_a
        T_init[1, 0] = sin_a
        T_init[1, 1] = cos_a
        
        reg_result = o3d.pipelines.registration.registration_icp(
            src_sweep, ref_sweep, max_corr_dist, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )
        
        if reg_result.fitness > best_fitness:
            best_fitness = reg_result.fitness
            best_T = reg_result.transformation
            
    print(f"  -> Best Z-rotation sweep fitness: {best_fitness:.4f}")
    return best_T



def run_icp_alignment(ref_pts, src_pts, init_s, init_R, init_t, voxel_size=0.05, scaling_refinement=False, use_global_reg=False):
    """
    Loads point cloud coords, downsamples them, and runs Open3D ICP.
    strictly acts as a refinement step starting from the initial alignment.
    
    scaling_refinement: If True, allows ICP to slightly refine the scale factor.
                         If False, scaling is disabled during ICP.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Error: Open3D is not installed. To run ICP, run 'pip install open3d' inside your conda environment.")
        sys.exit(1)
        
    print(f"Creating Open3D point clouds for ICP alignment")
    
    # Create Open3D PointCloud objects
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)
    
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pts)
    
    # Downsample point clouds for fast ICP
    print(f"Downsampling point clouds with voxel size: {voxel_size} m")
    ref_pcd_down = ref_pcd.voxel_down_sample(voxel_size)
    src_pcd_down = src_pcd.voxel_down_sample(voxel_size)
    
    # Estimate Normals for high-precision Point-to-Plane ICP
    print("Estimating normals for Point-to-Plane ICP...")
    ref_pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    src_pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # CRITICAL FIX: Pre-scale src_pcd_down before ICP. 
    # Open3D's Point-to-Plane ICP explodes if T_init contains scale.
    src_pcd_down.scale(init_s, center=(0,0,0))
    
    if use_global_reg:
        print("Computing normals and FPFH features for Global Registration...")
        
        ref_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            ref_pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
            
        distance_threshold = voxel_size * 1.5
        print("Running RANSAC Global Registration...")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd_down, ref_pcd_down, src_fpfh, ref_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        # Result transformation is rigid
        T_init = np.copy(result.transformation)
        print("Global Registration completed.")
    else:
        # Construct 4x4 initial transformation matrix (Rigid ONLY)
        T_init = np.eye(4)
        T_init[:3, :3] = init_R
        T_init[:3, 3] = init_t
    
    # Define max distance for registration (e.g. 5 times the voxel size)
    max_correspondence_distance = voxel_size * 5.0
    
    print("Running Open3D ICP registration...")
    # Select transformation estimation method
    if scaling_refinement:
        print("Using Point-to-Point ICP (Scaling Refinement Enabled)")
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    else:
        print("Using Point-to-Point ICP (Rigid Alignment)")
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
        
    reg_result = o3d.pipelines.registration.registration_icp(
        src_pcd_down, ref_pcd_down, max_correspondence_distance, T_init, estimation
    )
    
    T_final = reg_result.transformation
    print("ICP completed successfully.")
    
    # Extract scale, rotation, and translation from the 4x4 matrix T_final
    A = T_final[:3, :3]
    t = T_final[:3, 3]
    
    # Find extra scale factor s_icp if scaling_refinement was True
    s_icp = np.mean(np.linalg.norm(A, axis=0))
    
    # Extract clean rotation R
    R = A / s_icp
    # Orthogonalize R to remove numerical drift
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Total scale is init_s * s_icp
    s = init_s * s_icp
    
    return s, R, t


# ==========================================
# 3. PLY Rewriting Logic
# ==========================================

def apply_transformation_and_save(plydata, output_ply_path, s, R, t):
    """
    Applies (s, R, t) transformation to the source PLY and saves it.
    Preserves all points and custom attributes (zero data loss).
    """
    print(f"Applying transformation to 3DGS PLY: scale={s:.6f}, translation={t}")
    
    # 1. Transform positions (x, y, z)
    xyz = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    
    # Apply R, s, t: xyz_new = s * xyz * R.T + t
    xyz_new = s * (xyz @ R.T) + t
    
    plydata['vertex']['x'] = xyz_new[:, 0].astype(np.float32)
    plydata['vertex']['y'] = xyz_new[:, 1].astype(np.float32)
    plydata['vertex']['z'] = xyz_new[:, 2].astype(np.float32)
    
    # 2. Transform normals (nx, ny, nz) if they exist
    if 'nx' in plydata['vertex'] and 'ny' in plydata['vertex'] and 'nz' in plydata['vertex']:
        nx = plydata['vertex']['nx']
        ny = plydata['vertex']['ny']
        nz = plydata['vertex']['nz']
        normals = np.stack([nx, ny, nz], axis=1)
        # Normals are only rotated
        normals_new = normals @ R.T
        norms = np.linalg.norm(normals_new, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normals_new = normals_new / norms
        plydata['vertex']['nx'] = normals_new[:, 0].astype(np.float32)
        plydata['vertex']['ny'] = normals_new[:, 1].astype(np.float32)
        plydata['vertex']['nz'] = normals_new[:, 2].astype(np.float32)
        
    # 3. Scale log-scaling parameters (scale_0, scale_1, scale_2)
    # scale_i are stored in log space, so we add log(s)
    log_s = np.log(s)
    for i in range(3):
        scale_name = f'scale_{i}'
        if scale_name in plydata['vertex']:
            plydata['vertex'][scale_name] += log_s
            
    # 4. Rotate orientation parameters (rot_0, rot_1, rot_2, rot_3)
    # rot_0 is w, rot_1..3 are x, y, z
    if 'rot_0' in plydata['vertex']:
        q_R = rotation_matrix_to_quaternion(R)
        print(f"Applying orientation rotation using quaternion (w, x, y, z): {q_R}")
        
        q_old = np.stack([
            plydata['vertex']['rot_0'],
            plydata['vertex']['rot_1'],
            plydata['vertex']['rot_2'],
            plydata['vertex']['rot_3']
        ], axis=1)
        
        q_new = multiply_quaternions(q_R, q_old)
        
        plydata['vertex']['rot_0'] = q_new[:, 0].astype(np.float32)
        plydata['vertex']['rot_1'] = q_new[:, 1].astype(np.float32)
        plydata['vertex']['rot_2'] = q_new[:, 2].astype(np.float32)
        plydata['vertex']['rot_3'] = q_new[:, 3].astype(np.float32)
        
    # Write the modified PLY back
    os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
    print(f"Saving aligned PLY with 100% data preservation to: {output_ply_path}")
    plydata.write(output_ply_path)
    print(f"Transformation complete! Saved {len(xyz_new)} points.")


# ==========================================
# 4. Command Line Interface
# ==========================================

def parse_points(pts_str):
    """Parses points string formatted as: 'x1,y1,z1;x2,y2,z2;...'"""
    try:
        pts = []
        for p in pts_str.strip().split(';'):
            if p:
                pts.append([float(coord) for coord in p.split(',')])
        return np.array(pts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Points must be formatted as 'x1,y1,z1;x2,y2,z2;...': {e}")


def main():
    # Preprocess sys.argv to handle negative values starting with '-' when passed with space
    # e.g., convert ['--ref_bbox', '-9.377'] into ['--ref_bbox=-9.377']
    new_argv = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--') and i + 1 < len(sys.argv):
            next_arg = sys.argv[i + 1]
            if next_arg.startswith('-') and len(next_arg) > 1 and (next_arg[1].isdigit() or next_arg[1] == '.'):
                new_argv.append(f"{arg}={next_arg}")
                i += 2
                continue
        new_argv.append(arg)
        i += 1
    sys.argv = [sys.argv[0]] + new_argv

    parser = argparse.ArgumentParser(
        description="Regularize scale and align 3DGS point cloud to TLS reference."
    )
    
    parser.add_argument("--mode", required=True, choices=["bbox", "icp", "umeyama", "interactive", "auto"],
                        help="Alignment mode: 'bbox' for rough bounding box align, "
                             "'icp' for automatic ICP refinement, "
                             "'umeyama' for manual SVD point-picking via CLI args, "
                             "'interactive' to visually click points to align, "
                             "'auto' for fully automated robust clustering + Z-rotation sweep.")
    
    parser.add_argument("--ref_ply", required=True, help="Path to reference TLS PLY file.")
    parser.add_argument("--src_ply", required=True, help="Path to source 3DGS PLY file.")
    parser.add_argument("--output_ply", required=True, help="Path to save the aligned PLY file.")
    
    # Bbox mode arguments
    parser.add_argument("--ref_bbox", type=lambda s: [float(x) for x in s.split(',')],
                        help="Reference Bounding box: 'xmin,xmax,ymin,ymax' or 'xmin,xmax,ymin,ymax,zmin,zmax'. If omitted, computed automatically.")
    parser.add_argument("--src_bbox", type=lambda s: [float(x) for x in s.split(',')],
                        help="Source Bounding box: 'xmin,xmax,ymin,ymax' or 'xmin,xmax,ymin,ymax,zmin,zmax'. If omitted, computed automatically.")
    
    # ICP mode arguments
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size (meters) for downsampling point clouds before ICP (default: 0.05).")
    parser.add_argument("--refine_scale", action="store_true",
                        help="If set, ICP is allowed to slightly adjust the scale factor. "
                             "Otherwise, scaling is kept strictly fixed to the initial scale.")
    parser.add_argument("--global_reg", action="store_true",
                        help="Run Global Registration (FPFH + RANSAC) to find initial alignment instead of using Bounding Boxes. Highly recommended if orientations differ significantly.")
    
    # Umeyama mode arguments
    parser.add_argument("--ref_pts", type=parse_points,
                        help="Corresponding reference points in TLS coordinate frame: 'x1,y1,z1;x2,y2,z2;...'.")
    parser.add_argument("--src_pts", type=parse_points,
                        help="Corresponding source points in 3DGS coordinate frame: 'x1,y1,z1;x2,y2,z2;...'.")
                        
    args = parser.parse_args()
    
    print("\n[Progress] Loading PLY files into memory (this may take a moment)...")
    
    try:
        import open3d as o3d
        print(f"  -> Reading reference {args.ref_ply} via Open3D (memory optimized)...")
        ref_pcd = o3d.io.read_point_cloud(args.ref_ply)
        ref_pts = np.asarray(ref_pcd.points)
    except Exception as e:
        print(f"  -> Fallback to PlyData for ref_ply: {e}")
        ref_ply_data = PlyData.read(args.ref_ply)
        ref_pts = np.stack([ref_ply_data['vertex']['x'], ref_ply_data['vertex']['y'], ref_ply_data['vertex']['z']], axis=1)
        import open3d as o3d
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)

    print(f"  -> Reading source {args.src_ply} via plyfile (preserving attributes)...")
    src_ply_data = PlyData.read(args.src_ply)
    src_pts = np.stack([src_ply_data['vertex']['x'], src_ply_data['vertex']['y'], src_ply_data['vertex']['z']], axis=1)

    if args.mode in ["bbox", "icp"]:
        if args.global_reg and args.mode == "icp":
            print("Using Global Registration for initial alignment. Bounding boxes are ignored.")
        else:
            if not args.ref_bbox or not args.src_bbox:
                print("\n[Progress] Extracting main clusters to compute robust bounding boxes...")
                try:
                    import open3d as o3d
                    if not args.ref_bbox:
                        ref_pcd_temp = o3d.geometry.PointCloud()
                        ref_pcd_temp.points = o3d.utility.Vector3dVector(ref_pts)
                        ref_main = extract_main_cluster(ref_pcd_temp, args.voxel_size)
                        args.ref_bbox = get_pcd_bbox(ref_main)
                        print(f"Auto-computed Robust Reference BBox (from cluster):\n  {args.ref_bbox}")
                        
                    if not args.src_bbox:
                        src_pcd_temp = o3d.geometry.PointCloud()
                        src_pcd_temp.points = o3d.utility.Vector3dVector(src_pts)
                        src_main = extract_main_cluster(src_pcd_temp, args.voxel_size)
                        args.src_bbox = get_pcd_bbox(src_main)
                        print(f"Auto-computed Robust Source BBox (from cluster):\n  {args.src_bbox}")
                except Exception as e:
                    print(f"Clustering failed: {e}. Falling back to raw bounding boxes.")
                    if not args.ref_bbox:
                        args.ref_bbox = compute_bbox_from_pts(ref_pts)
                        print(f"Auto-computed Reference BBox (Raw):\n  {args.ref_bbox}")
                    if not args.src_bbox:
                        args.src_bbox = compute_bbox_from_pts(src_pts)
                        print(f"Auto-computed Source BBox (Raw):\n  {args.src_bbox}")

    if args.mode == "bbox":
        print("Mode: Bounding Box Alignment (Scale + Translation)")
        s, R, t = get_bbox_transform(args.ref_bbox, args.src_bbox)
        apply_transformation_and_save(src_ply_data, args.output_ply, s, R, t)
        
    elif args.mode == "icp":
        if args.global_reg:
            print("Mode: ICP Alignment Refinement (Using Global Registration for Initial Alignment)")
            
            try:
                import open3d as o3d
            except ImportError:
                parser.error("Open3D is required for clustering and global registration.")
                
            print("\n[Progress] Extracting main clusters to compute robust initial scale and translation...")
            
            ref_main = extract_main_cluster(ref_pcd, args.voxel_size)
            
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_pts)
            src_main = extract_main_cluster(src_pcd, args.voxel_size)
            
            ref_bbox = get_pcd_bbox(ref_main)
            src_bbox = get_pcd_bbox(src_main)
            
            print(f"Ref Main Cluster BBox:\n  {ref_bbox}")
            print(f"Src Main Cluster BBox:\n  {src_bbox}")
            
            # Compute translation and scale from bounding boxes of the main clusters
            init_s, _, init_t = get_bbox_transform(ref_bbox, src_bbox)
            init_R = np.eye(3)
            print(f"Estimated robust scale from main clusters = {init_s:.6f}")
        else:
            print("Mode: ICP Alignment Refinement (Using Bounding Box for Initial Alignment)")
            init_s, init_R, init_t = get_bbox_transform(args.ref_bbox, args.src_bbox)
            print(f"Estimated initial scale={init_s:.6f}, translation={init_t}")
            
        s, R, t = run_icp_alignment(
            ref_pts, src_pts, init_s, init_R, init_t,
            voxel_size=args.voxel_size,
            scaling_refinement=args.refine_scale,
            use_global_reg=args.global_reg
        )
        apply_transformation_and_save(src_ply_data, args.output_ply, s, R, t)
        
    elif args.mode == "umeyama":
        if args.ref_pts is None or args.src_pts is None:
            parser.error("umeyama mode requires both --ref_pts and --src_pts parameters.")
        if len(args.ref_pts) != len(args.src_pts):
            parser.error("Number of reference points must equal number of source points.")
            
        print(f"Mode: Umeyama SVD Point Alignment (Points Count: {len(args.ref_pts)})")
        s, R, t = umeyama(args.src_pts, args.ref_pts)
        
        print("\nCalculated Alignment Parameters:")
        print(f"Scale Factor (s): {s:.6f}")
        print("Rotation Matrix (R):\n", R)
        print(f"Translation Vector (t): {t}\n")
        
        apply_transformation_and_save(src_ply_data, args.output_ply, s, R, t)
        
    elif args.mode == "interactive":
        print("\nMode: Interactive Point-Picking Alignment")
        print("This will open a 3D viewer. You must pick the EXACT SAME 3 points in both point clouds.")
        
        try:
            import open3d as o3d
        except ImportError:
            parser.error("Open3D is required for interactive mode.")
            
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)
        # Downsample for visualization so it doesn't lag the viewer
        ref_pcd_vis = ref_pcd.voxel_down_sample(0.02)
        ref_pcd_vis.paint_uniform_color([0, 0.651, 0.929]) # Cyan
        
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_pts)
        src_pcd_vis = src_pcd.voxel_down_sample(0.02)
        src_pcd_vis.paint_uniform_color([1, 0.706, 0]) # Yellow
        
        picked_ref_pts = pick_points(ref_pcd_vis, "REFERENCE (TLS) - Pick 3 points")
        picked_src_pts = pick_points(src_pcd_vis, "SOURCE (3DGS) - Pick the SAME 3 points in the SAME ORDER")
        
        if len(picked_ref_pts) != len(picked_src_pts):
            print(f"Error: Point count mismatch! Reference: {len(picked_ref_pts)}, Source: {len(picked_src_pts)}")
            sys.exit(1)
            
        s, R, t = umeyama(picked_src_pts, picked_ref_pts)
        
        print("\nCalculated Alignment Parameters:")
        print(f"Scale Factor (s): {s:.6f}")
        print("Rotation Matrix (R):\n", R)
        print(f"Translation Vector (t): {t}\n")
        
        apply_transformation_and_save(src_ply_data, args.output_ply, s, R, t)
        
    elif args.mode == "auto":
        print("\nMode: Fully Automatic Alignment (Robust Clustering + Z-Rotation Sweep + ICP)")
        
        try:
            import open3d as o3d
        except ImportError:
            parser.error("Open3D is required for automatic mode.")
            
        print("\n[Progress] Extracting main clusters to compute robust bounding boxes...")
        
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_pts)
        
        ref_main = extract_main_cluster(ref_pcd, args.voxel_size)
        src_main = extract_main_cluster(src_pcd, args.voxel_size)
        
        ref_bbox = get_pcd_bbox(ref_main)
        src_bbox = get_pcd_bbox(src_main)
        print(f"  -> Robust Reference BBox: {ref_bbox}")
        print(f"  -> Robust Source BBox: {src_bbox}")
        
        # 1. BBox Transform (Scale + Translation)
        init_s, _, init_t = get_bbox_transform(ref_bbox, src_bbox)
        
        T_scale = np.eye(4)
        T_scale[0,0] = T_scale[1,1] = T_scale[2,2] = init_s
        
        T_trans1 = np.eye(4)
        T_trans1[:3, 3] = init_t
        
        # Apply scale and translation to a copy of src_main
        import copy
        src_main_aligned = copy.deepcopy(src_main)
        src_main_aligned.transform(T_trans1 @ T_scale)
        
        # 2. Translate to origin for rotation sweep
        ref_center = ref_main.get_center()
        
        T_to_origin = np.eye(4)
        T_to_origin[:3, 3] = -ref_center
        
        T_from_origin = np.eye(4)
        T_from_origin[:3, 3] = ref_center
        
        ref_main.transform(T_to_origin)
        src_main_aligned.transform(T_to_origin)
        
        # 3. Z-Rotation Sweep
        best_T_rot = auto_align_z_sweep(ref_main, src_main_aligned, args.voxel_size)
        
        # 4. Compose full initial transformation matrix
        T_total_init = T_from_origin @ best_T_rot @ T_to_origin @ T_trans1 @ T_scale
        
        # Extract s, R, t for ICP
        A = T_total_init[:3, :3]
        s_guess = np.mean(np.linalg.norm(A, axis=0))
        R_guess = A / s_guess
        
        # Orthogonalize R to be numerically safe
        U, _, Vt = np.linalg.svd(R_guess)
        R_guess = U @ Vt
        
        t_guess = T_total_init[:3, 3]
        
        print("\n[Progress] Running Final ICP Refinement...")
        s, R, t = run_icp_alignment(
            ref_pts, src_pts, s_guess, R_guess, t_guess,
            voxel_size=args.voxel_size,
            scaling_refinement=args.refine_scale,
            use_global_reg=False
        )
        
        apply_transformation_and_save(src_ply_data, args.output_ply, s, R, t)

if __name__ == "__main__":
    main()
