import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    Standard ResNet18 to extract a global feature vector from the X-ray.
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        # Use a lightweight ResNet
        resnet = models.resnet18(pretrained=True)
        
        # Change first layer to accept 1 channel (Grayscale) instead of 3 (RGB)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Project to desired dimension
        self.fc = nn.Linear(512, feature_dim)
        
    def forward(self, x):
        # x: (Batch, 1, 512, 512)
        features = self.backbone(x)        # (Batch, 512, 1, 1)
        features = features.view(features.size(0), -1) # Flatten
        return self.fc(features)           # (Batch, 256)

class GeometryEncoder(nn.Module):
    """
    PointNet-style encoder for the 3D centerline.
    Input: (Batch, N, 5) -> [x, y, z, u, v]
    """
    def __init__(self, input_dim=5, feature_dim=256):
        super().__init__()
        
        # Shared MLPs (applied to every point independently)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Final projection
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        # x: (Batch, N, 5)
        B, N, C = x.shape
        
        x_flat = x.view(-1, C)
        
        # Local Features
        feat_1 = self.mlp1(x_flat)  # (B*N, 128)
        feat_2 = self.mlp2(feat_1)  # (B*N, 512)
        
        # Reshape back to look for global max
        feat_2 = feat_2.view(B, N, 512)
        
        # Global Max Pooling (Key to PointNet)
        global_feat = torch.max(feat_2, 1)[0] # (B, 512)
        
        return self.fc(global_feat) # (B, 256)

class NeuroDeformNet(nn.Module):
    """
    The Main fusion Network.
    """
    def __init__(self, num_points=200):
        super().__init__()
        self.num_points = num_points
        
        # 1. Encoders
        self.img_enc = ImageEncoder(feature_dim=256)
        self.geo_enc = GeometryEncoder(input_dim=5, feature_dim=256)
        
        # 2. Decoder (Predicts displacement)
        # Input: ImageFeat(256) + GeoFeat(256) + PointCoord(3) = 515
        self.decoder = nn.Sequential(
            nn.Linear(256 + 256 + 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # Output: delta_x, delta_y, delta_z
        )

    def forward(self, image, geometry_input):
        """
        image: (B, 1, H, W)
        geometry_input: (B, N, 5) -> contains [x,y,z, u,v]
        """
        B, N, _ = geometry_input.shape
        
        # 1. Extract Global Features
        img_vec = self.img_enc(image)      # (B, 256)
        geo_vec = self.geo_enc(geometry_input) # (B, 256)
        
        # Concatenate Global Context
        global_vec = torch.cat([img_vec, geo_vec], dim=1) # (B, 512)
        
        # 2. Expand Global Vector to every point
        global_expanded = global_vec.unsqueeze(1).repeat(1, N, 1) # (B, N, 512)
        
        # Extract just the (x,y,z) part from input to help decoder
        original_xyz = geometry_input[:, :, :3]
        
        # Combine: [Global_Context, Local_XYZ]
        decoder_input = torch.cat([global_expanded, original_xyz], dim=2) # (B, N, 515)
        
        # 3. Predict Displacement
        decoder_input_flat = decoder_input.view(-1, 515)
        displacement_flat = self.decoder(decoder_input_flat) # (B*N, 3)
        
        # Reshape
        displacement = displacement_flat.view(B, N, 3)
        
        # 4. Final Output: Original + Delta
        predicted_centerline = original_xyz + displacement
        
        return predicted_centerline