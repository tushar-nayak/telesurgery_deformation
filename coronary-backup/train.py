import torch
from torch.utils.data import DataLoader
from dataset import VascularDataset
from model import NeuroDeformNet
from loss import NeuroVascularLoss
import os
import torch.optim as optim
from tqdm import tqdm

def train():
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4      # Small batch size for complex 3D data
    EPOCHS = 50         # Should see convergence quickly with clean synthetic data
    LR = 0.001          # Learning Rate
    DATA_DIR = "output_fluoroscopy_highres" # Point to your image folder
    
    print(f"Training on {DEVICE}...")

    # 1. Setup Data
    # Note: Ensure dataset.json is in DATA_DIR
    dataset = VascularDataset(data_dir=DATA_DIR, max_points=200)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Loaded {len(dataset)} training samples.")

    # 2. Setup Model & Loss
    model = NeuroDeformNet(num_points=200).to(DEVICE)
    loss_fn = NeuroVascularLoss(w_mse=1.0, w_curvature=0.2, w_length=0.05, w_proj=0.5)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            # Move data to GPU
            images = batch['image'].to(DEVICE)           # (B, 1, 512, 512)
            geo_input = batch['geometry_input'].to(DEVICE) # (B, N, 5)
            target_cl = batch['target_centerline'].to(DEVICE) # (B, N, 3)
            proj_mats = batch['projection_matrix'].to(DEVICE) # (B, 4, 4)
            
            # Forward Pass
            optimizer.zero_grad()
            predicted_cl = model(images, geo_input)
            
            # Calculate Loss (PINN Magic happens here)
            loss, loss_components = loss_fn(predicted_cl, target_cl, proj_mats)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "MSE": f"{loss_components['mse']:.4f}",
                "Phys": f"{loss_components['curvature']:.4f}"
            })
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.6f}")

        # Save Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint to checkpoints/model_epoch_{epoch+1}.pth")

    print("Training Complete!")

if __name__ == "__main__":
    train()