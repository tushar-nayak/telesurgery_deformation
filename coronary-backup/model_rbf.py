import torch
import torch.nn as nn
import torchvision.models as models

class DifferentiableRBFLayer(nn.Module):
    """
    Mathematical layer that converts Sparse Control Points -> Dense Centerline
    using Thin Plate Spline (Slide 10).
    No learnable weights; just pure physics.
    """
    def __init__(self):
        super().__init__()
        
    def phi(self, r):
        # Thin Plate Spline: r^2 * log(r)
        # Add epsilon to avoid log(0)
        return (r**2) * torch.log(r + 1e-6)

    def forward(self, sparse_disp, original_cp, original_dense):
        """
        sparse_disp: (B, M, 3) -> Predicted forces
        original_cp: (B, M, 3) -> Locations of control points
        original_dense: (B, N, 3) -> Locations of all points
        """
        # We need to solve for weights 'w' such that Phi * w = sparse_disp
        # But for simplified deformation (assuming RBF places standard kernels), 
        # we can compute the interpolation matrix.
        
        # 1. Compute Distance Matrix between Dense points and Control Points
        # Dense (N, 1, 3) - CP (1, M, 3) -> (N, M, 3)
        diff = original_dense.unsqueeze(2) - original_cp.unsqueeze(1)
        dists = torch.norm(diff, dim=3) # (B, N, M)
        
        # 2. Apply Kernel
        K = self.phi(dists) # (B, N, M)
        
        # 3. Normalize K (Simple RBF interpolation often solves a linear system,
        # but for a neural net generator, we can treat sparse_disp as the 'weights'
        # directly for stability, or we can approximate).
        # Let's treat predicted sparse_disp as the WEIGHTS of the RBF kernels.
        # Dense_Disp = Sum(Weight_i * Phi(||x - c_i||))
        
        # (B, N, M) @ (B, M, 3) -> (B, N, 3)
        dense_disp = torch.bmm(K, sparse_disp)
        
        return dense_disp

class NeuroDeformRBF(nn.Module):
    def __init__(self, num_control_points=10, num_dense_points=200):
        super().__init__()
        self.M = num_control_points
        self.N = num_dense_points
        
        # 1. Encoders (Same as before)
        self.img_enc = models.resnet18(pretrained=True)
        self.img_enc.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.img_enc.fc = nn.Linear(512, 256)
        
        self.geo_enc = nn.Sequential(
            nn.Linear(3, 64), # Input is just (x,y,z) of base shape
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        # PointNet maxpool
        
        # 2. Fusion & Parameter Prediction
        # We only predict M vectors (10 * 3 = 30 values), not 200 * 3
        self.fc_fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, self.M * 3) # Output: Sparse Control Vectors
        )
        
        # 3. Physics Layer
        self.rbf_layer = DifferentiableRBFLayer()

    def forward(self, image, base_dense_xyz, base_cp_xyz):
        """
        image: (B, 1, 512, 512)
        base_dense_xyz: (B, 200, 3) - The undeformed line
        base_cp_xyz: (B, 10, 3) - The undeformed control points
        """
        B = image.shape[0]
        
        # Visual Features
        img_feat = self.img_enc(image) # (B, 256)
        
        # Geometry Features (Simple PointNet)
        geo_feat = self.geo_enc(base_dense_xyz) # (B, N, 256)
        geo_feat = torch.max(geo_feat, 1)[0]    # (B, 256) Global Max Pool
        
        # Fuse
        global_feat = torch.cat([img_feat, geo_feat], dim=1)
        
        # Predict SPARSE Parameters
        sparse_disp_flat = self.fc_fusion(global_feat)
        sparse_disp = sparse_disp_flat.view(B, self.M, 3) # (B, 10, 3)
        
        # Compute DENSE Field via Physics
        # Note: We scale down the RBF output significantly because TPS kernel values can be huge
        dense_disp = self.rbf_layer(sparse_disp, base_cp_xyz, base_dense_xyz)
        
        # Normalize RBF magnitude (heuristic for stability)
        dense_disp = dense_disp * 0.01 
        
        # Final Shape
        predicted_dense = base_dense_xyz + dense_disp
        
        return sparse_disp, predicted_dense