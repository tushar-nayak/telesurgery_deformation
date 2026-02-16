#!/usr/bin/env python3

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

class XRayDepthGenerator:
    def __init__(self, name, tag):
        self.name = name
        self.meshname = name + ".stl"
        self.current_dir = os.getcwd()
        self.mesh_file = os.path.join(self.current_dir, "meshes", self.meshname)
        self.tag = tag
        
        # Load the mesh file
        if not os.path.exists(self.mesh_file):
            raise IOError(f"Mesh file not found: {self.mesh_file}")

        print(f"Loading mesh from {self.mesh_file}")
        self.mesh = trimesh.load(self.mesh_file, process=False)  # Load the mesh without processing

        centroid_offset = self.mesh.bounding_box.centroid

        self.mesh.apply_translation(-centroid_offset)

        print("Mesh loaded!")


    def apply_rotation(self, rotation_matrix):
        print("Rotation matrix: ", rotation_matrix)
        self.mesh.apply_transform(rotation_matrix)
        self.rotated_mesh_file = os.path.join(os.getcwd(), "meshes", f"{self.name}_{self.tag}.stl")
        self.mesh.export(self.rotated_mesh_file)


    def get_xray_and_depth(self):
         # ==============gvxr setup===================
        # Create an OpenGL context
        # print("Create an OpenGL context")
        gvxr.createOpenGLContext()


        # Create a source
        # print("Set up the beam")
        gvxr.setSourcePosition(-40.0,  0.0, 0.0, "cm")
        gvxr.usePointSource()

        # Set its spectrum, here a monochromatic beam
        # 1000 photons of 80 keV (i.e. 0.08 MeV) per ray
        gvxr.setMonoChromatic(0.08, "MeV", 1000)
        # The following is equivalent: gvxr.setMonoChromatic(80, "keV", 1000)

        # Set up the detector
        # print("Set up the detector")
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
        if not os.path.exists(self.rotated_mesh_file):
            raise IOError(self.rotated_mesh_file)

        print("Load the mesh data from", self.rotated_mesh_file)
        gvxr.loadMeshFile("vessel", self.rotated_mesh_file, "mm")


        # print("Move ", "vessel", " to the centre")
        gvxr.moveToCentre("vessel")
        recentering_offet = gvxr.getNodeWorldTransformationMatrix("vessel")
        # print("location of mesh after centering", gvxr.getNodeWorldTransformationMatrix("vessel"))


        # Material properties
        # print("Set ", "vessel", "'s material")

        # Liquid water
        gvxr.setCompound("vessel", "H2O")
        gvxr.setDensity("vessel", 0.3, "g/cm3")
        gvxr.setDensity("vessel", 0.3, "g.cm-3")


        # Compute an X-ray image
        # We convert the array in a Numpy structure and store the data using single-precision floating-point numbers.
        # print("Compute an X-ray image")
        x_ray_image = np.array(gvxr.computeXRayImage()).astype(np.single)
        # print("X-ray image shape:", x_ray_image.shape)
        # Save the X-ray image using PIL
        xray_image_uint8 = (255 * (x_ray_image - np.min(x_ray_image)) / (np.max(x_ray_image) - np.min(x_ray_image))).astype(np.uint8)
        xray_image_pil = PIL.Image.fromarray(xray_image_uint8)
        xray_image_pil.save(f"output_data/{self.name}_xray_{self.tag}_image.png")


        # Update the visualisation window
        gvxr.displayScene()

        # Create the output directory if needed
        if not os.path.exists("output_data"):
            os.mkdir("output_data")

        # Save the L-buffer
        gvxr.saveLastLBuffer('output_data/lbuffer-02.tif')

        # Display the X-ray image
        # using a linear colour scale
        if has_mpl:
            plt.figure(figsize=(10, 5))
            plt.title("Image simulated using gVirtualXray\nusing a linear colour scale")
            plt.imshow(x_ray_image, cmap="gray")
            plt.colorbar(orientation='vertical')
            print("Showing xray image...")
            plt.savefig(f"output_data/{self.name}_xray_image_{self.tag}_graph.png", dpi=300)
            plt.show()
            

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
            plt.imsave(f"output_data/{self.name}_{self.tag}_screenshot.png", np.array(screenshot))




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
        if not os.path.exists(self.rotated_mesh_file):
            raise IOError(f"Mesh file not found: {self.rotated_mesh_file}")

        print(f"Loading mesh from {self.rotated_mesh_file}")
        mesh = trimesh.load(self.rotated_mesh_file, process=False)  # load 'as-is', without processing

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

        # print("source_rh", source_rh)
        # print("detector_rh", detector_rh)

        #  ray grid based on detector config
        num_pixels = (640, 320)
        spacing = (0.5, 0.5)  # in mm

        # detector_rh is the center of the detector plane
        detector_center = detector_rh

        # Create grid axes
        grid_x = (np.arange(num_pixels[0]) - num_pixels[0] / 2) * spacing[0]
        grid_y = -(np.arange(num_pixels[1]) - num_pixels[1] / 2) * spacing[1]
        xx, yy = np.meshgrid(grid_x, grid_y)

        # Compute detector plane axes 
        # Compute the ray direction (from source to detector) to serve as the plane normal.
        detector_normal = trimesh.unitize(detector_center - source_rh)
        # Use the up-vector that was set in GVXR; negated since GVXR uses a left-handed coordinate system
        up_rh = np.array([0, 0, -1])
        # Ensure the up vector isn’t parallel to the normal:
        if np.allclose(np.cross(up_rh, detector_normal), 0):
            up_rh = np.array([0, 1, 0])
            
        # print("up_rh", up_rh)
        # print("detector_normal", detector_normal)
        x_axis = trimesh.unitize(np.cross(up_rh, detector_normal))
        y_axis = trimesh.unitize(np.cross(detector_normal, x_axis))

        # print("x_axis", x_axis)
        # print("y_axis", y_axis)


        # Build pixel positions on the detector plane 
        pixel_positions = detector_center + xx[..., None] * x_axis + yy[..., None] * y_axis

        # Compute ray origins and directions
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
        print("Showing depth image...")
        plt.savefig(f"output_data/{self.name}_{self.tag}_depth_graph.png", dpi=300)
        plt.show()

        gvxr.renderLoop()