import os
import subprocess
import shutil
import glob

base_dir = "d:/Chirayu/NIR"
workspace_dir = f"{base_dir}/colmap_workspace_lab2"
images_dir = f"{workspace_dir}/images"
db_path = f"{workspace_dir}/database.db"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if os.path.exists(f"{workspace_dir}/sparse_abs_pose"):
    shutil.rmtree(f"{workspace_dir}/sparse_abs_pose")
os.makedirs(f"{workspace_dir}/sparse_abs_pose", exist_ok=True)

colmap_bins = glob.glob("d:/Chirayu/colmap_bin/**/COLMAP.bat", recursive=True)
colmap_bin = colmap_bins[0]

print("Running COLMAP mapper (Lowered Abs Pose)...")
run_cmd(f"{colmap_bin} mapper --database_path \"{db_path}\" --image_path \"{images_dir}\" --output_path \"{workspace_dir}/sparse_abs_pose\" --Mapper.abs_pose_min_num_inliers 15 --Mapper.abs_pose_min_inlier_ratio 0.1")

print(f"Done. Check the output in {workspace_dir}/sparse_abs_pose/0")
