write me a latex report, of what I have done till now from the day one

/home/intern/LAI/Report

you have to add images that are saved in this folder

I need this report to be in details

ok, 

started with:

first understanding of the 3D Gaussian Splatting for Real-Time Radiance Field Rendering paper, and Estimating the leaf area index from terrestrial laser scanning point cloud data with fractal theory

problem statement: we are trying to achieve a good lai estimation using 3DGS with comparison to true LAI (TLS LAI) , as 3dgs is very cheap and comparitively less time consuming than TLS

deciding that to keep FDTLS tool as the estimation baseline

finding and understanding why we are keeping FDTLS tool as the baseline

learning and understanding how LIDAR sensors work, and learning to operate FARO TLS sensor

after understanding 3dgs, it's repo was cloned and I tried to implement it, but faced a lot of issues when dealing with the instalations of the libriries (the main reason, why the repo was not working properly, was because 3dgs works on older versions of the librires used, thus leading to version issue) 

then

data collection, I tried to collect the data of various plants (indoor as well as outdoor plants), as it was my first time using the instuments, I made mistakes, which lead to bad data, like distancing between the tls sensor and the plant being either too less  (leading to cutting of the imp plant data) and being too much (which lead to the plant data being sparse), or taking the data at a higher resolution leading to multiple crashes in the preprocessing app

learning and understanding how to use the preprocessing app for FARO TLS (SCENE), and the doing the preprocessing and segmentation of multiple trees

making videos using the dji camera and then turn them into 3dgs

how does 3dgs work?

1. input video -> extract frames using ffmpeg
2. mapping these frames together into a point cloud using COLMAP
3. using train.py of the 3dgs to generate the gaussians 

issues faced here, where the different formats of the two outputs, and unknowingly downsampling the point clouds leading to bad results during the lai estimation

after all the processes, getting .ply files of both tls and 3dgs, comparing them using fdtls tool, getting bad results

reasons due to which the lai estimation of 3dgs was wrong, 1. Scale mismatch (the tls point cloud is in meters, i.e world coorinate scale, the 3dgs was in arbitary scale)

2. Floaters in 3dgs 

how the issues were resolved?

for floaters, i did data post processing on 3dgs point cloud, where, I applied opacity filter to remove the gaussians created out of thin air by 3dgs, as it has no sense of depth

and i also applied SOR, for the removal of the floaters

I also explored a few models that are based on floater removal on 3dgs, like StableGS, GaussionFusion and 3DGS Simplify

and I tried using 3dgs simplify, and it works great at removing floaters, but it downsamples the point cloud

another problem faced during the data collection for the videos for 3dgs is movement of leaves, which is destorying the structure if the 3dgs, making it unusable for the lai estimation, this is a dynamic problem, which I currently dont have a solution to, and I will be working on it later, that's why, for now I shifted my focus from outdoor plants to indoor plants as the leaves dont move as much indoors, these indoor plants will be used to tackle the problem of scale mismatching

today's task was to work my way thorugh the scale mismatch:

issues faced: the two .ply cloud were not overlapping properly, as they should have had, so I first tried aligning the point clouds using 

bounding boxes alignment, which was then refined by icp, but it failed, reason: segmentation is different for both of the plant point clouds

then I tried mapping similar data point clusters together:

how did i implement it?

1. sor
2. DBSCAN clustering
3. robust bounding box
4. global regiration and ICP

this failed as well, why? too few correspondences after mutual fitler, this was because global reg. uses FPFH (Fast pt. feature histograms) but as plants have a lot of nearly identical leaves, the algo gets confused.

so I changed my apporach to manual registration and it worked out really good, I alomost perfectly overlapped the two point clouds

now my focus is to try doing reg. using ICP, but I am hitting a wall here

update:
finally broke through the wall! the manual alignment was too tedious and not reproducible, so we built a fully automated pipeline for alignment and scale regularization.

the new automated pipeline works as follows:
1. voxel downsampling to standardize point density
2. Statistical Outlier Removal (SOR) to denoise
3. DBSCAN clustering to extract the main plant structure and discard floaters
4. Principal Component Analysis (PCA) to extract the principal axes and bounding extents
5. Automatic Isotropic Scaling: we estimate the scale factor by taking the median ratio of the PCA standard deviations. this perfectly scales the 3dgs cloud to the metric TLS cloud.
6. Z-Rotation Sweep & ICP: because PCA can have eigenvector flips, we test all 180-degree flips and refine them using Point-to-Plane ICP. We then select the best alignment using a composite score (Fitness / RMSE).

results of the automated pipeline:
the automated PCA-ICP pipeline works flawlessly! for example, on the Outdoor Small Plant, the F-score jumped to 99.08%, and on Tree 5 it hit 99.75%.

post-processing ablation and LAI discoveries:
we then ran our post-processing (SOR) on the auto-aligned clouds and calculated the FDTLS LAI. this revealed a massive scientific finding: geometry accuracy and ecological metric accuracy are tightly correlated!
- for datasets where floaters were safely removed without destroying canopy (like Tree 5 and Outdoor Small Plant), the pure geometric F-score improved, and the relative LAI error dropped massively (Tree 5 LAI error dropped from 38.18% to 5.33%).
- conversely, for datasets where the filtering was too aggressive and accidentally removed legitimate thin canopy structures (like Tree 1 and Bushes), the geometric F-score worsened, and consequently, the LAI error worsened.

this proves that FDTLS is highly sensitive to physical occupancy, and accurate LAI estimation strictly requires high-fidelity, scale-accurate geometric reconstruction rather than mere visual rendering quality.

the final paper (updated_paper.tex) has been fully updated with these findings, including empirical tables for alignment metrics, geometry comparisons, and FDTLS LAI relative errors.