import open3d as o3d
import numpy as np
import copy
import os
import torch
import torch.nn.functional as F

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # cyan
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def save_merged_point_cloud(source, target, transformation, filename):
    source_copy = copy.deepcopy(source)
    source_copy.transform(transformation)
    # source_copy.paint_uniform_color([0.5, 0.5, 0.5])  # Grey
    target_copy = copy.deepcopy(target)
    merged = source_copy + target_copy
    
    # Save PLY
    o3d.io.write_point_cloud(filename, merged)
    print(f"Saved merged point cloud to: {filename}")

    # Save .bin
    points = np.asarray(merged.points, dtype=np.float32)  # Nx3
    filename_bin = filename.replace(".ply", ".bin")
    points.tofile(filename_bin)
    print(f"Saved merged point cloud to: {filename_bin}")   

# def chamfer_distance(p1, p2, chunk_size=1000):
#     with torch.no_grad():
#         dist1_idx = []
#         for i in range(0, p1.size(0), chunk_size):
#             chunk = p1[i:i+chunk_size]
#             dist1_idx.append(torch.cdist(chunk, p2).argmin(dim=1))
#         dist1_idx = torch.cat(dist1_idx)
        
#         dist2_idx = []
#         for i in range(0, p2.size(0), chunk_size):
#             chunk = p2[i:i+chunk_size]
#             dist2_idx.append(torch.cdist(chunk, p1).argmin(dim=1))
#         dist2_idx = torch.cat(dist2_idx)
        
#     closest_in_p2 = p2[dist1_idx]
#     dist1 = torch.norm(p1 - closest_in_p2, dim=1)
    
#     closest_in_p1 = p1[dist2_idx]
#     dist2 = torch.norm(p2 - closest_in_p1, dim=1)
    
#     return dist1.mean() + dist2.mean()

# =========================================================
# Chamfer Distance
# =========================================================
def chamfer_distance(p1, p2, chunk_size=1000, return_tensor=True):

    with torch.no_grad():

        dist1_idx = []

        for i in range(0, p1.size(0), chunk_size):

            chunk = p1[i:i + chunk_size]

            dist1_idx.append(torch.cdist(chunk, p2).argmin(dim=1))

        dist1_idx = torch.cat(dist1_idx)

        dist2_idx = []

        for i in range(0, p2.size(0), chunk_size):

            chunk = p2[i:i + chunk_size]

            dist2_idx.append(torch.cdist(chunk, p1).argmin(dim=1))

        dist2_idx = torch.cat(dist2_idx)

    closest_in_p2 = p2[dist1_idx]
    dist1 = torch.norm(p1 - closest_in_p2, dim=1)

    closest_in_p1 = p1[dist2_idx]
    dist2 = torch.norm(p2 - closest_in_p1, dim=1)

    cd = dist1.mean() + dist2.mean()

    if return_tensor:
        return cd
    else:
        return cd.item()


# =========================================================
# Compute Metrics
# =========================================================
def compute_metrics(source, target, transformation):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_copy = copy.deepcopy(source)
    source_copy.transform(transformation)

    source_np = np.asarray(source_copy.points)
    target_np = np.asarray(target.points)

    source_tensor = torch.from_numpy(source_np).float().to(device)
    target_tensor = torch.from_numpy(target_np).float().to(device)

    # Chamfer Distance
    cd = chamfer_distance(
        source_tensor,
        target_tensor,
        return_tensor=False
    )

    # ICP Evaluation Metrics
    threshold = 0.2

    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_copy,
        target,
        threshold
    )

    fitness = evaluation.fitness
    rmse = evaluation.inlier_rmse

    return cd, fitness, rmse

def optimize_with_chamfer(source_np, target_np, init_transform, steps=300, lr=1e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_tensor = torch.from_numpy(source_np).float().to(device)
    target_tensor = torch.from_numpy(target_np).float().to(device)

    transform = torch.eye(4, requires_grad=True, device=device)
    transform.data = torch.from_numpy(init_transform).float().to(device)

    optimizer = torch.optim.Adam([transform], lr=lr)

    for i in range(steps):
        R = transform[:3, :3]
        t = transform[:3, 3]
        transformed_source = torch.matmul(source_tensor, R.T) + t
        loss = chamfer_distance(transformed_source, target_tensor, return_tensor = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"[{i}/{steps}] Chamfer Loss: {loss.item():.6f}")

    return transform.detach().cpu().numpy()

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

if __name__ == "__main__":
    print("Loading point clouds...")
    source = o3d.io.read_point_cloud("/home/intern/LAI/Final_PCD/tree1_tls.ply")

    target = o3d.io.read_point_cloud("//home/intern/LAI/Final_PCD/tree1_rgb_3dgs.ply")

    # Auto-calculate voxel size relative to object scale (avoids manual tuning)
    bbox = target.get_axis_aligned_bounding_box()
    voxel_size = max(bbox.get_extent()) / 100.0  

    # === 1. Global Registration (Fixes "opposite/misaligned" initial state) ===
    print("\n=== Global Registration (FPFH + RANSAC) ===")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    global_reg = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("Global Registration Transformation:\n", global_reg.transformation)

    # === 2. Local Registration (Point-to-Plane ICP) ===
    print("\n=== Local Registration (Point-to-Plane ICP) ===")
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    threshold = voxel_size * 0.4
    # Note: We pass global_reg.transformation as the initial guess here!
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, global_reg.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print("ICP Transformation:\n", reg_p2l.transformation)

    # === 3. Chamfer Distance Optimization (Deep Learning refinement) ===
    print("\n=== Chamfer Distance Optimization ===")
    # Optionally downsample before Chamfer if points are very dense to save memory/time. 
    # But for maximum fidelity, we keep the original points:
    source_np = np.asarray(source.points)
    target_np = np.asarray(target.points)

    refined_trans = optimize_with_chamfer(source_np, target_np, reg_p2l.transformation, steps=500)
    print("Refined Transformation:\n", refined_trans)

    # =====================================================
    # Compute Metrics
    # =====================================================
    cd, fitness, rmse = compute_metrics(
        source,
        target,
        refined_trans
    )

    print("\n==============================")
    print("Registration Metrics")
    print("==============================")

    print(f"Chamfer Distance : {cd:.6f}")
    print(f"Fitness Score    : {fitness:.6f}")
    print(f"Inlier RMSE      : {rmse:.6f}")

    # ✅ Save Results
    np.savetxt("refined_transform.txt", refined_trans, fmt="%.8f")
    print("Saved refined transformation to: refined_transform.txt")

    print("\nFinal Alignment Result:")
    draw_registration_result(source, target, refined_trans)

    save_merged_point_cloud(source, target, refined_trans, "merged_chamfer_refined.ply")