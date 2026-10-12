@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set TORCH_CUDA_ARCH_LIST=8.6
call conda activate gaussian_splatting
python "D:\Chirayu\gaussian-splatting\render_comparison.py" -s "D:\Chirayu\NIR\lab2_undistorted" -m "D:\Chirayu\gaussian-splatting\output\nir_tree_features" --iteration 5000
