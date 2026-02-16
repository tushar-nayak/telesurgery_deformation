Creating the environment
conda env create -f environment.yml
conda activate endovascular

Files in this project:

-- Helper Files --
xray_depth_generator.py - Class definition derived from the gvxr documentation that generates a depth and simulated xray/fluoroscopic image based on the input mesh and homogeneous transform.

centerline_extraction2.py - helper file that skeletonizes the 3D meshes using the skeletor library, also derived from their documentation. This has already been documentation

skeletonizer_2d.py - class definition that accepts a 2D image and determines the branching and endpoints of the vessel, then generates a skeleton (networkx graph). This is done ahead of time as it takes a while to run.

combined_2.py - the original (less clean) version of xray_depth_generator.py, which was used to generate the target (rotated) xray and depth images


-- Scripts to Run --

1. Spectral analysis
spectral analysis.py - extracts graphs from both the 2D image and 3D mesh such that only terminal and branch points are pulled, then applies a spectral embedding, which is then used to solve the Hungarian problem. This essentially determines which nodes are most likely corresponding between the two graphs. Then this is applied to a RANSAC-PnP solver to generate the estimated transformation. The estimated 

to run: python spectral_analysis.py
It will show the binarized image of the input fluoroscopic scan, then the skeletonized version, then the extracted graph. Occasionally, it will stop after this due to the PnP solver failing to generate a valid solution. If that is the case, just run it again. Afterward, it will show the (estimated) transformed version of the anatomy superimposed on the original fluoroscopic image. Then it will show an image of the transformed anatomy as a fluoroscopic image (saved to output_data/03-b_xray_image_spectral_graph) and as a depth image (saved to output_data/03-b_spectral_depth_graph).The depth image takes a minute or two to generate. Lastly, you will see an interactive window showing the xray setup. It will also print the estimated rotation matrix and the dice score.



2. Endpoint Search
endpoint_registration.py - extracts the terminal points only from both 2D and the 3D data. For the 3D data, it is projected to 2D and then endpoints are extracted according to the process done for 2D. Then it searches across all possible combinations of 4 points, which are then processed using RANSAC PnP. The best is determined based on the greatest dice score. 

to run: python endpoint_registration.py - It will show the binarized image of the input fluoroscopic scan and  the skeletonized version. Then it will show the  the binarized image of the 3D data projected to 2D followed by the skeletonized version. It will then execute the search process, which takes a few minutes. Each time an improved set is found, the best dice score will be printed.Afterward, it will show the (estimated) transformed version of the anatomy superimposed on the original fluoroscopic image. Then it will show an image of the transformed anatomy as a fluoroscopic image (saved to output_data/03-b_xray_image_endpoint_graph) and as a depth image (saved to output_data/03-b_endpoint_depth_graph). The depth image takes a minute or two to generate. Lastly, you will see an interactive window showing the xray setup. It will also print the estimated rotation matrix and the dice score.

3. Global Search
global_search.py - genrates rotation matrices uniformly about all three axes in 30 degree increments, then applies these rotations and projects the resulting shape to 2D. This search continues until the configuration with the best dice score if found. 

to run: python global_search.py - It will begin the seach (WARNING: This can take hours!), then once completed, it will show the  (estimated) transformed version of the anatomy superimposed on the original fluoroscopic image. Then it will show an image of the transformed anatomy as a fluoroscopic image (saved to output_data/03-b_xray_image_global_graph) and as a depth image (saved to output_data/03-b_global_depth_graph).The depth image takes a minute or two to generate. Lastly, you will see an interactive window showing the xray setup. It will also print the estimated rotation matrix and the dice score.


Note: There were a few different iterations of these methods that took inspiration from each other (eg. global search followed by terminal point RANSAC PnP), but the codebase got a bit messy so I didn't include them here. I also assessed other metrics that looked both at region overlap and pointcloud methods, but I ended up choosing dice scores