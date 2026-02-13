import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import cv2
from torchvision import transforms

class VascularDataset(Dataset):
    def __init__(self, data_dir, image_size=512, max_points=200):
        """
        Args:
            data_dir: Folder containing 'dataset_labels.json' (e.g., output_fluoroscopy_highres)
            image_size: Target size for the CNN (e.g., 512x512)
            max_points: Fixed number of points for the centerline (needed for batching)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_points = max_points
        
        # 1. Load the Master Index
        # We try to find the json inside the data_dir
        json_path = os.path.join(data_dir, "dataset.json")
        
        # Fallback: In previous scripts, we might have named it 'dataset_labels.json'
        if not os.path.exists(json_path):
            json_path = os.path.join(data_dir, "dataset_labels.json")
            
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"CRITICAL: Could not find 'dataset.json' or 'dataset_labels.json' in {data_dir}")
            
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
            
        # Create a list of keys (filenames) to index into
        self.keys = list(self.labels.keys())
        
        # 2. Define Image Normalization
        # Converts 0-255 image to 0-1 Tensor, then normalizes to [-1, 1] range
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

    def __len__(self):
        return len(self.keys)

    def resample_centerline(self, points):
        """
        Resamples a curve to have exactly 'max_points'.
        This is REQUIRED because PyTorch cannot batch tensors of different sizes.
        """
        if len(points) == self.max_points:
            return points
            
        # Calculate cumulative distance along the line
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_len = cumulative_dist[-1]
        
        # Create new evenly spaced distances
        new_dists = np.linspace(0, total_len, self.max_points)
        
        # Interpolate x, y, z coordinates
        new_points = np.zeros((self.max_points, 3))
        for i in range(3):
            new_points[:, i] = np.interp(new_dists, cumulative_dist, points[:, i])
            
        return new_points

    def project_points(self, points_3d, proj_mat):
        """
        Projects 3D points to 2D using the camera matrix.
        This creates the "Hint" feature (u, v) for the network.
        """
        # Add homogeneous coord (x, y, z, 1)
        ones = np.ones((len(points_3d), 1))
        points_h = np.hstack((points_3d, ones)) # (N, 4)
        
        # Apply Matrix: (N, 4) @ (4, 4).T
        projected = points_h @ proj_mat.T 
        
        # Extract U, V
        # Note: If simulating perspective, you would divide by Z here.
        # For our affine approximation/orthographic setup, direct U,V is fine.
        u = projected[:, 0]
        v = projected[:, 1]
        
        # Normalize roughly to [-1, 1] range to help the neural network
        # Assuming field of view is roughly 200mm
        u_norm = u / 100.0 
        v_norm = v / 100.0
        
        return np.stack([u_norm, v_norm], axis=1)

    def __getitem__(self, idx):
        # 1. Get Entry
        key = self.keys[idx]
        entry = self.labels[key]
        
        # 2. Load Image
        img_path = os.path.join(self.data_dir, key)
        if not os.path.exists(img_path):
             # Try looking in current directory if absolute path fails
             img_path = key 
             
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # Resize to target (e.g., 512x512)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image_tensor = self.transform(image) # (1, 512, 512)
        
        # 3. Load Physics Data (Centerlines)
        # The JSON has the path to the .npy file
        cl_path = entry['centerline_file']
        
        if not os.path.exists(cl_path):
            raise FileNotFoundError(f"Centerline not found: {cl_path}")
            
        # Load the "Ground Truth" deformed shape
        deformed_cl = np.load(cl_path).astype(np.float32)
        
        # 4. Generate "Input" Shape (The Undeformed / Reference)
        # Ideally, you load the 'original' pre-op centerline here.
        # Since we are training a simulator, we can simulate the "Pre-op"
        # by taking the Deformed and adding noise or smoothing it.
        # This teaches the network: "Here is a rough guess, fix it to match the image."
        noise = np.random.normal(0, 1.0, deformed_cl.shape).astype(np.float32)
        original_cl = deformed_cl + noise
        
        # 5. Resample both to fixed size
        original_cl = self.resample_centerline(original_cl)
        deformed_cl = self.resample_centerline(deformed_cl)
        
        # 6. Feature Engineering
        # Get the camera matrix
        proj_mat = np.array(entry['projection_matrix'], dtype=np.float32)
        
        # Project the "Original" points to 2D to create hints
        proj_2d = self.project_points(original_cl, proj_mat) # (N, 2)
        
        # Create Input Vector: (x, y, z, u, v)
        geometry_input = np.concatenate([original_cl, proj_2d], axis=1).astype(np.float32)

        return {
            "image": image_tensor,                   # Input Image
            "geometry_input": geometry_input,        # Input Shape + Hints
            "target_centerline": deformed_cl,        # Ground Truth Shape
            "projection_matrix": proj_mat            # Camera Physics
        }