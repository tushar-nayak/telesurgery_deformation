import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from dataset import VascularDataset
from model import NeuroDeformNet
import numpy as np
import os

def create_rotation_video():
    # --- CONFIG ---
    CHECKPOINT = "checkpoints/model_epoch_50.pth" # Ensure this matches your best epoch
    DATA_DIR = "output_fluoroscopy_highres"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data & Model
    print("Loading resources...")
    ds = VascularDataset(data_dir=DATA_DIR, max_points=200)
    model = NeuroDeformNet(num_points=200).to(DEVICE)
    
    # Load Weights
    if not os.path.exists(CHECKPOINT):
        # Auto-find latest
        checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pth")])
        CHECKPOINT = os.path.join("checkpoints", checkpoints[-1])
    
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # 2. Pick a good sample (Try a few until you find a complex one)
    idx = np.random.randint(0, len(ds))
    sample = ds[idx]
    
    # Prepare Tensors
    image = sample['image'].unsqueeze(0).to(DEVICE)
    geo = torch.from_numpy(sample['geometry_input']).unsqueeze(0).to(DEVICE)
    target = sample['target_centerline']

    # 3. Predict
    with torch.no_grad():
        prediction = model(image, geo)
        pred_numpy = prediction.cpu().squeeze(0).numpy()

    # 4. Create Animation
    print(f"Generating video for Sample {idx}...")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        # Plot Red Prediction
        ax.scatter(pred_numpy[:,0], pred_numpy[:,1], pred_numpy[:,2], c='red', s=30, label='AI Reconstruction')
        # Plot Blue Truth (faint)
        ax.scatter(target[:,0], target[:,1], target[:,2], c='blue', s=10, alpha=0.1)
        
        ax.set_title(f"AI Reconstruction (Rotating View) - Frame {frame}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend()
        
        # Rotate the camera
        ax.view_init(elev=20, azim=frame)
        return fig,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    
    # Save
    save_path = "reconstruction_demo.gif"
    print(f"Saving to {save_path}...")
    ani.save(save_path, writer='pillow', fps=20)
    print("Done! Open 'reconstruction_demo.gif' to see your AI in action.")

if __name__ == "__main__":
    create_rotation_video()