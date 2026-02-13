import torch
from model import NeuroDeformNet
from dataset import VascularDataset
from torch.utils.data import DataLoader
import os

def test_pipeline():
    print("--- STARTING SANITY CHECK ---")
    
    # 1. Check Data Directory
    data_dir = "output_fluoroscopy_highres" # Or 'final_dataset' depending on your naming
    if not os.path.exists(data_dir):
        print(f"ERROR: Could not find folder '{data_dir}'.")
        print("Please check your folder name.")
        return

    print(f"1. Loading Dataset from '{data_dir}'...")
    try:
        # Initialize Dataset
        ds = VascularDataset(data_dir=data_dir, max_points=200)
        loader = DataLoader(ds, batch_size=2, shuffle=True)
        
        # Grab a single batch
        batch = next(iter(loader))
        imgs = batch['image']
        geo = batch['geometry_input']
        target = batch['target_centerline']
        proj = batch['projection_matrix']
        
        print(f"   [Success] Batch Loaded!")
        print(f"   - Images:   {imgs.shape}  (Should be [2, 1, 512, 512])")
        print(f"   - Geometry: {geo.shape}    (Should be [2, 200, 5])")
        print(f"   - Target:   {target.shape}    (Should be [2, 200, 3])")
        
    except Exception as e:
        print(f"   [FAILED] Dataset Error: {e}")
        return

    # 2. Check Model
    print("\n2. Initializing Neural Network...")
    try:
        model = NeuroDeformNet(num_points=200)
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        imgs = imgs.to(device)
        geo = geo.to(device)
        
        print(f"   [Success] Model created on {device}.")
        
    except Exception as e:
        print(f"   [FAILED] Model Error: {e}")
        return

    # 3. Run Forward Pass
    print("\n3. Running Forward Pass (Prediction)...")
    try:
        prediction = model(imgs, geo)
        print(f"   [Success] Forward pass complete!")
        print(f"   - Output Shape: {prediction.shape} (Should be [2, 200, 3])")
        
        # Simple Loss Check
        loss = torch.nn.functional.mse_loss(prediction, target.to(device))
        print(f"   - Dummy Loss Value: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   [FAILED] Forward Pass Error: {e}")
        return

    print("\n--- SANITY CHECK PASSED ---")
    print("You are ready to run 'python train.py'")

if __name__ == "__main__":
    test_pipeline()