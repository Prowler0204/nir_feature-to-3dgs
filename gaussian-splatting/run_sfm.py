import os
import subprocess
import shutil
import glob

base_dir = "d:/Chirayu/NIR"
workspace_dir = f"{base_dir}/colmap_workspace_lab2"
images_dir = f"{workspace_dir}/images"
sparse_dir = f"{workspace_dir}/sparse/0"
db_path = f"{workspace_dir}/database.db"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)



# Clean workspace for forced mapper output
if os.path.exists(f"{workspace_dir}/sparse_forced"):
    shutil.rmtree(f"{workspace_dir}/sparse_forced")
os.makedirs(f"{workspace_dir}/sparse_forced", exist_ok=True)

colmap_bins = glob.glob("d:/Chirayu/colmap_bin/**/COLMAP.bat", recursive=True)
if not colmap_bins:
    print("Could not find COLMAP.bat")
    exit(1)
colmap_bin = colmap_bins[0]
print(f"Found COLMAP at: {colmap_bin}")

# 3. Run Pipeline (Mapper Only)
print("Running COLMAP mapper (Forced Seed)...")
run_cmd(f"{colmap_bin} mapper --database_path \"{db_path}\" --image_path \"{images_dir}\" --output_path \"{workspace_dir}/sparse_forced\" --Mapper.init_image_id1 177 --Mapper.init_image_id2 46 --Mapper.init_min_tri_angle 8 --Mapper.min_model_size 10")

print(f"Done. Check the output in {workspace_dir}/sparse_forced/0")
