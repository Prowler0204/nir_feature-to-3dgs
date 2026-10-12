#!/usr/bin/env python3
"""
Interactive Manual Alignment Tool for 3DGS and TLS
This tool places the 3DGS and TLS point clouds side-by-side.
You can manually adjust the SCALE and ORIENTATION of the 3DGS cloud until it matches the TLS cloud.
Translation is ignored/bypassed as per user request.

Controls:
  W / S : Increase / Decrease Scale
  Up / Down : Rotate Pitch (X-axis)
  Left / Right : Rotate Yaw (Z-axis)
  Q / E : Rotate Roll (Y-axis)
  Enter : Save and Exit
  Esc : Exit without saving
"""

import open3d as o3d
import numpy as np
import argparse
import sys
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

def apply_transformation_and_save(src_ply_data, output_ply_path, scale, R):
    print("\nApplying transformation to original 3DGS PLY...")
    print(f"Scale: {scale:.6f}")
    
    vertex_data = src_ply_data.elements[0].data
    
    # Extract positions
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    pts = np.vstack((x, y, z)).T
    
    # 1. Zero-center original points before rotation/scaling
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    
    # 2. Scale and Rotate
    pts_transformed = scale * (pts_centered @ R.T)
    
    # Update positions
    vertex_data['x'] = pts_transformed[:, 0]
    vertex_data['y'] = pts_transformed[:, 1]
    vertex_data['z'] = pts_transformed[:, 2]
    
    # Update 3DGS Scales
    if 'scale_0' in vertex_data.dtype.names:
        log_s = np.log(scale)
        vertex_data['scale_0'] += log_s
        vertex_data['scale_1'] += log_s
        vertex_data['scale_2'] += log_s
        
    # Update 3DGS Quaternions
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
    
    print(f"Saving fully preserved aligned PLY to: {output_ply_path}")
    new_plydata.write(output_ply_path)
    print("Done!")

class ManualAligner:
    def __init__(self, ref_pcd, src_pcd, src_ply_data, output_path):
        self.ref_pcd = ref_pcd
        self.src_pcd = src_pcd
        self.src_ply_data = src_ply_data
        self.output_path = output_path
        
        # We need a copy of the source to apply cumulative transforms
        self.src_pcd_original = copy.deepcopy(src_pcd)
        
        self.current_scale = 1.0
        self.current_R = np.eye(3)
        
        # Center both at origin to rotate around their own centers
        self.ref_centroid = self.ref_pcd.get_center()
        self.src_centroid = self.src_pcd_original.get_center()
        
        self.ref_pcd.translate(-self.ref_centroid)
        self.src_pcd_original.translate(-self.src_centroid)
        
        # Move them side by side
        # TLS on the left, 3DGS on the right
        self.offset = np.array([self.ref_pcd.get_max_bound()[0] * 2.0, 0, 0])
        self.ref_pcd.translate(-self.offset)
        self.src_pcd_original.translate(self.offset)
        
        # Initialize viewing cloud
        self.src_pcd.points = self.src_pcd_original.points
        self.src_pcd.colors = self.src_pcd_original.colors
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Manual 3DGS Aligner", width=1280, height=720)
        
        self.vis.add_geometry(self.ref_pcd)
        self.vis.add_geometry(self.src_pcd)
        
        self.register_callbacks()
        
        print("\n--- CONTROLS ---")
        print("W / S        : Scale Up / Scale Down")
        print("Up / Down    : Rotate Pitch (X-axis)")
        print("Left / Right : Rotate Yaw (Z-axis)")
        print("Q / E        : Rotate Roll (Y-axis)")
        print("Enter        : Save and Exit")
        print("Esc          : Exit without saving\n")

    def update_geometry(self):
        # Apply current scale and rotation to original
        pts = np.asarray(self.src_pcd_original.points)
        
        # Translate to origin of its local space
        pts_centered = pts - self.offset
        
        # Rotate and Scale
        pts_transformed = self.current_scale * (pts_centered @ self.current_R.T)
        
        # Translate back
        pts_final = pts_transformed + self.offset
        
        self.src_pcd.points = o3d.utility.Vector3dVector(pts_final)
        self.vis.update_geometry(self.src_pcd)

    def scale_up(self, vis):
        self.current_scale *= 1.02
        self.update_geometry()
        return False
        
    def scale_down(self, vis):
        self.current_scale *= 0.98
        self.update_geometry()
        return False

    def rot_x_pos(self, vis):
        R = self.src_pcd.get_rotation_matrix_from_xyz((np.radians(2), 0, 0))
        self.current_R = R @ self.current_R
        self.update_geometry()
        return False
        
    def rot_x_neg(self, vis):
        R = self.src_pcd.get_rotation_matrix_from_xyz((np.radians(-2), 0, 0))
        self.current_R = R @ self.current_R
        self.update_geometry()
        return False

    def rot_y_pos(self, vis):
        R = self.src_pcd.get_rotation_matrix_from_xyz((0, np.radians(2), 0))
        self.current_R = R @ self.current_R
        self.update_geometry()
        return False
        
    def rot_y_neg(self, vis):
        R = self.src_pcd.get_rotation_matrix_from_xyz((0, np.radians(-2), 0))
        self.current_R = R @ self.current_R
        self.update_geometry()
        return False

    def rot_z_pos(self, vis):
        R = self.src_pcd.get_rotation_matrix_from_xyz((0, 0, np.radians(2)))
        self.current_R = R @ self.current_R
        self.update_geometry()
        return False
        
    def rot_z_neg(self, vis):
        R = self.src_pcd.get_rotation_matrix_from_xyz((0, 0, np.radians(-2)))
        self.current_R = R @ self.current_R
        self.update_geometry()
        return False

    def save_and_exit(self, vis):
        apply_transformation_and_save(self.src_ply_data, self.output_path, self.current_scale, self.current_R)
        self.vis.destroy_window()
        sys.exit(0)
        return False

    def register_callbacks(self):
        self.vis.register_key_callback(ord('W'), self.scale_up)
        self.vis.register_key_callback(ord('S'), self.scale_down)
        
        self.vis.register_key_callback(265, self.rot_x_pos) # Up arrow
        self.vis.register_key_callback(266, self.rot_x_neg) # Down arrow
        
        self.vis.register_key_callback(262, self.rot_z_pos) # Right arrow
        self.vis.register_key_callback(263, self.rot_z_neg) # Left arrow
        
        self.vis.register_key_callback(ord('Q'), self.rot_y_pos)
        self.vis.register_key_callback(ord('E'), self.rot_y_neg)
        
        self.vis.register_key_callback(257, self.save_and_exit) # Enter

    def run(self):
        self.vis.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual scale/rotation alignment tool")
    parser.add_argument("--ref_ply", required=True, help="Path to TLS reference .ply")
    parser.add_argument("--src_ply", required=True, help="Path to 3DGS source .ply")
    parser.add_argument("--output_ply", required=True, help="Path to save aligned 3DGS .ply")
    args = parser.parse_args()

    print("Loading point clouds...")
    ref_pcd = o3d.io.read_point_cloud(args.ref_ply)
    src_pcd = o3d.io.read_point_cloud(args.src_ply)
    
    # Downsample for smooth interaction
    ref_pcd = ref_pcd.voxel_down_sample(voxel_size=0.02)
    src_pcd = src_pcd.voxel_down_sample(voxel_size=0.02)
    
    ref_pcd.paint_uniform_color([0, 0.8, 0]) # Green
    src_pcd.paint_uniform_color([1, 0, 0]) # Red

    print("Reading original 3DGS attributes via plyfile...")
    src_ply_data = PlyData.read(args.src_ply)

    aligner = ManualAligner(ref_pcd, src_pcd, src_ply_data, args.output_ply)
    aligner.run()
