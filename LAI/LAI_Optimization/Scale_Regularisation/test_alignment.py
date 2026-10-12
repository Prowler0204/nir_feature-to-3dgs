#!/usr/bin/env python3
"""
Unit test for align_3dgs_to_tls.py
Generates a mock PLY file with a single Gaussian, applies a known transformation,
and verifies that coordinates, log-scales, and quaternions are transformed correctly.
"""

import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement

# Import functions from align_3dgs_to_tls
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from align_3dgs_to_tls import (
    rotation_matrix_to_quaternion,
    multiply_quaternions,
    apply_transformation_and_save
)

def create_mock_ply(filename):
    """Creates a mock 3DGS PLY with 1 point."""
    # Define fields
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    # 1 point: pos=(1.0, 2.0, 3.0), log_scale=(0.5, 0.5, 0.5), rot=(1.0, 0.0, 0.0, 0.0) [identity]
    element_data = np.array([
        (1.0, 2.0, 3.0, 0.8, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3)
    ], dtype=dtype)
    
    el = PlyElement.describe(element_data, 'vertex')
    PlyData([el]).write(filename)
    print(f"Created mock PLY: {filename}")


def run_test():
    mock_in = "mock_input.ply"
    mock_out = "mock_output.ply"
    
    create_mock_ply(mock_in)
    
    # Define a known transformation:
    # Scale: s = 2.0
    # Rotation: 90 degrees around Z axis
    # Translation: t = [10.0, -5.0, 2.0]
    s = 2.0
    theta = np.pi / 2.0
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    t = np.array([10.0, -5.0, 2.0])
    
    apply_transformation_and_save(mock_in, mock_out, s, R, t)
    
    # Verify the output
    plydata = PlyData.read(mock_out)
    vertex = plydata['vertex']
    
    x, y, z = vertex['x'][0], vertex['y'][0], vertex['z'][0]
    s0, s1, s2 = vertex['scale_0'][0], vertex['scale_1'][0], vertex['scale_2'][0]
    r0, r1, r2, r3 = vertex['rot_0'][0], vertex['rot_1'][0], vertex['rot_2'][0], vertex['rot_3'][0]
    opacity = vertex['opacity'][0]
    
    # Expected Position:
    # p' = s * R * p + t
    # p' = 2.0 * [ -2, 1, 3 ]^T + [10, -5, 2]^T = [6.0, -3.0, 8.0]^T
    expected_pos = np.array([6.0, -3.0, 8.0])
    actual_pos = np.array([x, y, z])
    
    # Expected log scale:
    # scale' = scale + log(s) = 0.5 + log(2.0) = 0.5 + 0.69314718 = 1.19314718
    expected_scale = 0.5 + np.log(2.0)
    actual_scale = np.array([s0, s1, s2])
    
    # Expected quaternion:
    # Rotation quaternion for 90 deg around Z is:
    # q_R = [cos(45 deg), 0, 0, sin(45 deg)] = [0.70710678, 0, 0, 0.70710678]
    # Multiply by identity q_old = [1, 0, 0, 0] gives q_R.
    expected_q = np.array([np.cos(np.pi/4.0), 0.0, 0.0, np.sin(np.pi/4.0)])
    actual_q = np.array([r0, r1, r2, r3])
    
    print("\n--- Verification Results ---")
    print(f"Position: Expected {expected_pos}, Actual {actual_pos}")
    print(f"Log-scale: Expected {expected_scale:.6f}, Actual {actual_scale}")
    print(f"Quaternion: Expected {expected_q}, Actual {actual_q}")
    print(f"Opacity (should be unchanged): Expected 0.8, Actual {opacity}")
    
    assert np.allclose(actual_pos, expected_pos, atol=1e-5), "Position mismatch!"
    assert np.allclose(actual_scale, expected_scale, atol=1e-5), "Scale mismatch!"
    assert np.allclose(actual_q, expected_q, atol=1e-5) or np.allclose(actual_q, -expected_q, atol=1e-5), "Quaternion mismatch!"
    assert np.isclose(opacity, 0.8), "Opacity modified!"
    
    print("\nSUCCESS: All transformations verified successfully! Math is 100% correct.")
    
    # Clean up
    if os.path.exists(mock_in): os.remove(mock_in)
    if os.path.exists(mock_out): os.remove(mock_out)

if __name__ == "__main__":
    run_test()
