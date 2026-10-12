import os
import subprocess

datasets = [
    ("Bushes", "bushes.ply", "bushes_3dgs.ply"),
    ("Outdoor Small Plant", "outdoor_small_plant.ply", "outdoor_small_plant_3dgs.ply"),
    ("Tree 1", "tree1_tls.ply", "tree1_rgb_3dgs.ply"),
    ("Tree 5", "indoor_tree5_wp.ply", "indoor_tree5_3dgs_wp.ply")
]

for name, ref, src in datasets:
    print(f"--- {name} ---")
    cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python",
        "/home/intern/LAI/LAI_Optimization/Scale_Regularisation/auto_align_improved.py",
        "--ref_ply", f"/home/intern/LAI/Final_PCD/{ref}",
        "--src_ply", f"/home/intern/LAI/Final_PCD/{src}",
        "--output_ply", "/tmp/dummy.ply"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if "Fitness :" in line or "RMSE    :" in line or "Scale   :" in line:
            print(line.strip())
