#!/usr/bin/env python3


#use this one!!

# Import packages
import os
import numpy as np
import PIL.Image
import trimesh

has_mpl = True
try:
	import matplotlib # To plot images
	import matplotlib.pyplot as plt # Plotting
	from matplotlib.colors import LogNorm # Look up table
	from matplotlib.colors import PowerNorm # Look up table

	font = {'family' : 'serif',
			 'size'   : 15
		   }
	matplotlib.rc('font', **font)

except:
	has_mpl = False

# try:
# 	import trimesh.ray.ray_pyembree
# 	print("Using pyembree for ray tracing.")
# except ImportError:
# 	print("pyembree not installed. Falling back to default ray tracing engine.")

from gvxrPython3 import gvxr # Simulate X-ray images



name = "03-a"
meshname = name + ".stl"
current_dir = os.getcwd()
old_mesh_file = os.path.join(current_dir, "meshes", meshname)


#================= for setting up targets===================
tag = "target"

# Load the mesh file
if not os.path.exists(old_mesh_file):
	raise IOError(f"Mesh file not found: {old_mesh_file}")

print(f"Loading mesh from {old_mesh_file}")
mesh = trimesh.load(old_mesh_file, process=False)  # Load the mesh without processing

centroid_offset = mesh.bounding_box.centroid

mesh.apply_translation(-centroid_offset)

# Apply a random rotation
# random_rotation = trimesh.transformations.random_rotation_matrix()
random_rotation = np.array([[0.707, -0.707, 0, 0],
			[0.707, 0.707, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
])

print("Rotation matrix: ", random_rotation)
mesh.apply_transform(random_rotation)

# mesh.apply_translation(centroid_offset)


# Save the rotated mesh
mesh_file = os.path.join(current_dir, "meshes", f"{name}_{tag}.stl")
mesh.export(mesh_file)
print(f"{tag} mesh saved to {mesh_file}")
# ================= for setting up targets===================


# ==================for applying transforms from ransac method===================
# tag = "estimate_global"

# # Load the mesh file
# if not os.path.exists(old_mesh_file):
# 	raise IOError(f"Mesh file not found: {old_mesh_file}")

# print(f"Loading mesh from {old_mesh_file}")
# mesh = trimesh.load(old_mesh_file, process=False)  # Load the mesh without processing



# # # spectral output
# # estimated_rotation = np.array([
# # 	[ 0.03988947, -0.7165997,   0.69634309, 0],
# # 	[ 0.72178313,  0.50258309,  0.47585644, 0],
# # 	[-0.69096885,  0.48362703,  0.53727734, 0],
# # 	[0, 0, 0, 1]
# # ])

# # endpoint output
# # estimated_rotation = np.array([
# #  [ 0.69638165, -0.70646923, -0.12630844, 0],
# #  [ 0.67344015,  0.70409352, -0.22523471, 0],
# #  [ 0.24805435,  0.07178814,  0.96608256, 0],
# #  [0, 0, 0, 1]
# # ])

# # clobal output
# # estimated_rotation = np.array([
# #  [ 0.8660254, -0.5      ,  0.       , 0],
# #  [ 0.5      ,  0.8660254,  0.       , 0],
# #  [ 0.       ,  0.       ,  1.       , 0],
# #  [0, 0, 0, 1]
# # ])


# mesh.apply_transform(estimated_rotation)

# # Save the rotated mesh
# mesh_file = os.path.join(current_dir, "meshes", f"{name}_{tag}.stl")
# mesh.export(mesh_file)
# print(f"{tag} mesh saved to {mesh_file}")

# ==================for applying transforms from ransac method===================

# ==================for applying transforms from ransac method===================
# tag = "original"

# # Load the mesh file
# if not os.path.exists(old_mesh_file):
# 	raise IOError(f"Mesh file not found: {old_mesh_file}")

# mesh_file = old_mesh_file
# print(f"Loading mesh from {old_mesh_file}")
# mesh = trimesh.load(old_mesh_file, process=False)  # Load the mesh without processing

# ==================for applying transforms from ransac method===================




# ==============gvxr setup===================
# Create an OpenGL context
print("Create an OpenGL context")
gvxr.createOpenGLContext()


# Create a source
print("Set up the beam")
gvxr.setSourcePosition(-40.0,  0.0, 0.0, "cm")
gvxr.usePointSource()

# Set its spectrum, here a monochromatic beam
# 1000 photons of 80 keV (i.e. 0.08 MeV) per ray
gvxr.setMonoChromatic(0.08, "MeV", 1000)
# The following is equivalent: gvxr.setMonoChromatic(80, "keV", 1000)

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(10.0, 0.0, 0.0, "cm")
gvxr.setDetectorUpVector(0, 0, 1)
gvxr.setDetectorNumberOfPixels(640, 320)
gvxr.setDetectorPixelSize(0.5, 0.5, "mm")

# Calculate the center of the detector
detector_position = np.array([10.0, 0.0, 0.0])
detector_up_vector = np.array([0,0,1])
detector_pixel_size = np.array((0.5, 0.5)) # in mm
detector_dimensions = np.array((640,320))

# Load the sample data
if not os.path.exists(mesh_file):
	raise IOError(mesh_file)

print("Load the mesh data from", mesh_file)
gvxr.loadMeshFile("vessel", mesh_file, "mm")


# print("Move ", "vessel", " to the centre")
gvxr.moveToCentre("vessel")
recentering_offet = gvxr.getNodeWorldTransformationMatrix("vessel")
print("location of mesh after centering", gvxr.getNodeWorldTransformationMatrix("vessel"))


# Material properties
print("Set ", "vessel", "'s material")

# Liquid water
gvxr.setCompound("vessel", "H2O")
gvxr.setDensity("vessel", 0.3, "g/cm3")
gvxr.setDensity("vessel", 0.3, "g.cm-3")


# Compute an X-ray image
# We convert the array in a Numpy structure and store the data using single-precision floating-point numbers.
print("Compute an X-ray image")
x_ray_image = np.array(gvxr.computeXRayImage()).astype(np.single)
print("X-ray image shape:", x_ray_image.shape)
# Save the X-ray image using PIL
xray_image_uint8 = (255 * (x_ray_image - np.min(x_ray_image)) / (np.max(x_ray_image) - np.min(x_ray_image))).astype(np.uint8)
xray_image_pil = PIL.Image.fromarray(xray_image_uint8)
xray_image_pil.save(f"output_data/{name}_xray_{tag}_image.png")


# Update the visualisation window
gvxr.displayScene()

# Create the output directory if needed
if not os.path.exists("output_data"):
	os.mkdir("output_data")

# Save the X-ray image in a TIFF file and store the data using single-precision floating-point numbers.
# gvxr.saveLastXRayImage('output_data/raw_x-ray_image-02.tif')

# The line below will also works
# imwrite('output_data/raw_x-ray_image-02.tif', x_ray_image)

# Save the L-buffer
gvxr.saveLastLBuffer('output_data/lbuffer-02.tif')

# Display the X-ray image
# using a linear colour scale
if has_mpl:
	plt.figure(figsize=(10, 5))
	plt.title("Image simulated using gVirtualXray\nusing a linear colour scale")
	plt.imshow(x_ray_image, cmap="gray")
	plt.colorbar(orientation='vertical')
	# plt.show()
	plt.savefig(f"output_data/{name}_xray_image_{tag}_graph.png", dpi=300)
	

# Change the sample's colour
# By default the object is white, which is not always pretty. Let's change it to purple.
red = 102 / 255
green = 51 / 255
blue = 153 / 255
gvxr.setColour("vessel", red, green, blue, 1.0)

# This image can be used in a research paper to illustrate the simulation environment, in which case you may want to change the background colour to white with:
gvxr.setWindowBackGroundColour(1.0, 1.0, 1.0)

# Update the visualisation window
gvxr.displayScene()

# Take the screenshot and save it in a file
if has_mpl:
	screenshot = gvxr.takeScreenshot()
	plt.imsave(f"output_data/{name}_{tag}_screenshot.png", np.array(screenshot))

	# or display it using Matplotlib
	# plt.figure(figsize=(10, 10))
	# plt.imshow(screenshot)
	# plt.title("Screenshot of the X-ray simulation environment")
	# plt.axis('off')
	# plt.show()




# Interactive visualisation
# The user can rotate the 3D scene and zoom-in and -out in the visualisation window.

# - Keys are:
#     - Q/Escape: to quit the event loop (does not close the window)
#     - B: display/hide the X-ray beam
#     - W: display the polygon meshes in solid or wireframe
#     - N: display the X-ray image in negative or positive
#     - H: display/hide the X-ray detector
# - Mouse interactions:
#     - Zoom in/out: mouse wheel
#     - Rotation: Right mouse button down + move cursor```

#==================depth setup===================

# --- Load Mesh with Trimesh ---
if not os.path.exists(mesh_file):
    raise IOError(f"Mesh file not found: {mesh_file}")

print(f"Loading mesh from {mesh_file}")
mesh = trimesh.load(mesh_file, process=False)  # load 'as-is', without processing

# Instead of centering by the vertex centroid, center by the bounding box centroid
mesh.apply_translation(-mesh.bounding_box.centroid)

# --- Convert from GVXR left-handed to Trimesh right-handed coordinates ---
# Define transformation from LH to RH (flip Z-axis)
LH_to_RH = np.array([
    [1, 0,  0, 0],
    [0, 1,  0, 0],
    [0, 0, -1, 0],
    [0, 0,  0, 1]
])

# Get GVXR transformation matrix (from moveToCentre) and convert it:
transform_matrix_lh = np.array(gvxr.getNodeWorldTransformationMatrix("vessel"))
transform_matrix_rh = LH_to_RH @ transform_matrix_lh @ LH_to_RH
mesh.apply_transform(transform_matrix_rh)

# --- Convert GVXR source and detector positions from LH to RH ---
source_lh = np.array(gvxr.getSourcePosition("mm"))
source_rh = np.array([source_lh[0], source_lh[1], -source_lh[2]])

detector_lh = np.array(gvxr.getDetectorPosition("mm"))
detector_rh = np.array([detector_lh[0], detector_lh[1], -detector_lh[2]])

print("source_rh", source_rh)
print("detector_rh", detector_rh)

# --- Setup ray grid (using manually set pixel spacing) ---
num_pixels = (640, 320)
spacing = (0.5, 0.5)  # in mm

# Use detector_rh as the center of the detector plane
detector_center = detector_rh

# Create grid axes
grid_x = (np.arange(num_pixels[0]) - num_pixels[0] / 2) * spacing[0]
grid_y = -(np.arange(num_pixels[1]) - num_pixels[1] / 2) * spacing[1]
# Optionally, if flipping is needed, adjust here; for now, try without extra negatives.
xx, yy = np.meshgrid(grid_x, grid_y)

# --- Compute detector plane axes ---
# Compute the ray direction (from source to detector) to serve as the plane normal.
detector_normal = trimesh.unitize(detector_center - source_rh)
# Use the up-vector that was set in GVXR; since you set (0,0,1) in GVXR (LH) it becomes (0,0,-1) in RH
up_rh = np.array([0, 0, -1])
# Ensure the up vector isn’t parallel to the normal:
if np.allclose(np.cross(up_rh, detector_normal), 0):
    up_rh = np.array([0, 1, 0])
    
print("up_rh", up_rh)
print("detector_normal", detector_normal)
# Define axes: x_axis for horizontal, y_axis for vertical
x_axis = trimesh.unitize(np.cross(up_rh, detector_normal))
y_axis = trimesh.unitize(np.cross(detector_normal, x_axis))

print("x_axis", x_axis)
print("y_axis", y_axis)


# --- Build pixel positions on the detector plane ---
pixel_positions = detector_center + xx[..., None] * x_axis + yy[..., None] * y_axis

# --- Compute ray origins and directions ---
ray_origins = np.full(pixel_positions.shape, source_rh)
ray_directions = pixel_positions - ray_origins
ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)

# Flatten for raycasting:
ray_origins = ray_origins.reshape(-1, 3)
ray_directions = ray_directions.reshape(-1, 3)

# Perform ray-mesh intersection:
locations, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins=ray_origins,
    ray_directions=ray_directions,
    multiple_hits=False
)

# --- Create depth image ---
depth_image = np.full(ray_origins.shape[0], np.nan)
depths = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
depth_image[index_ray] = depths
depth_image = depth_image.reshape(num_pixels[1], num_pixels[0])

# Normalize and save the depth image:
depth_image_uint8 = (255 * (depth_image - np.nanmin(depth_image)) / 
                       (np.nanmax(depth_image) - np.nanmin(depth_image))).astype(np.uint8)
plt.figure(figsize=(10, 5))
plt.title("Depth Image")
plt.imshow(depth_image_uint8, cmap="viridis")
plt.colorbar(orientation='vertical')
plt.savefig(f"output_data/{name}_{tag}_depth_graph.png", dpi=300)
# depth_image_pil = PIL.Image.fromarray(depth_image_uint8)
# depth_image_pil.save(f"output_data/{name}_{tag}_depth.png")

gvxr.renderLoop()