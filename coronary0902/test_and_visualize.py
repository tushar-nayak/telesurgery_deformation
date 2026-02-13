import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import VascularDataset
from model import NeuroDeformNet
import os
import numpy as np

def visualize_prediction():
    # --- CONFIG ---
    TARGET_CHECKPOINT = "checkpoints/model_epoch_50.pth" 
    DATA_DIR = "output_fluoroscopy_highres"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- 3D RECONSTRUCTION VISUALIZER (SCATTER MODE) ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Could not find {DATA_DIR}")
        return
    
    ds = VascularDataset(data_dir=DATA_DIR, max_points=200)
    idx = np.random.randint(0, len(ds))
    sample = ds[idx]
    
    image = sample['image'].unsqueeze(0).to(DEVICE)
    geo = torch.from_numpy(sample['geometry_input']).unsqueeze(0).to(DEVICE)
    target = sample['target_centerline']
    
    # 2. Load Model
    checkpoint_path = TARGET_CHECKPOINT
    if not os.path.exists(checkpoint_path):
        # Fallback to latest
        if os.path.exists("checkpoints"):
            checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pth")])
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                checkpoint_path = os.path.join("checkpoints", checkpoints[-1])
                print(f"  (Using latest checkpoint: {checkpoint_path})")
            else:
                return

    model = NeuroDeformNet(num_points=200).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    # 3. Predict
    print("Running Inference...")
    with torch.no_grad():
        prediction = model(image, geo)
        pred_numpy = prediction.cpu().squeeze(0).numpy()

    # 4. Visualization (SCATTER MODE)
    print("Plotting...")
    fig = plt.figure(figsize=(14, 6))
    
    # --- View 1: 3D Point Cloud ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # GROUND TRUTH = BLUE DOTS
    ax1.scatter(target[:,0], target[:,1], target[:,2], c='blue', s=20, alpha=0.3, label='Ground Truth')
    
    # PREDICTION = RED DOTS
    ax1.scatter(pred_numpy[:,0], pred_numpy[:,1], pred_numpy[:,2], c='red', s=25, label='AI Prediction')
    
    ax1.set_title(f"3D Point Cloud (Sample {idx})")
    ax1.legend()
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    
    # --- View 2: X-Ray ---
    ax2 = fig.add_subplot(1, 2, 2)
    img_disp = sample['image'].squeeze().numpy()
    img_disp = (img_disp * 0.5) + 0.5
    ax2.imshow(img_disp, cmap='gray')
    ax2.set_title("Input Fluoroscopy")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Done!")

if __name__ == "__main__":
    visualize_prediction()