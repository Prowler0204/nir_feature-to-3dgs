@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set TORCH_CUDA_ARCH_LIST=8.6
call conda activate gaussian_splatting
python "D:\Chirayu\gaussian-splatting\train.py" -s "D:\Chirayu\NIR\lab2_undistorted" -m "D:\Chirayu\gaussian-splatting\output\nir_tree_features" --iterations 30000 --save_iterations 5000 10000 15000 20000 25000 30000
