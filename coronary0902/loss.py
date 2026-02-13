import torch
import torch.nn as nn

class NeuroVascularLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_curvature=0.1, w_length=0.01, w_proj=0.5):
        """
        w_mse: Weight for direct 3D coordinate matching (Supervised)
        w_curvature: Penalty for sharp bends (Stiffness)
        w_length: Penalty for uneven point spacing (Stretching)
        w_proj: Penalty for 2D visual misalignment
        """
        super().__init__()
        self.w_mse = w_mse
        self.w_curvature = w_curvature
        self.w_length = w_length
        self.w_proj = w_proj
        
        self.mse = nn.MSELoss()

    def curvature_loss(self, points):
        """
        Calculates the 2nd derivative (discrete Laplacian) of the curve.
        Minimizing this makes the curve smoother (resists bending).
        Formula: |p(i+1) - 2p(i) + p(i-1)|^2
        """
        # Force float32
        points = points.float()
        
        # Shift points to get neighbors
        p_left = points[:, :-2, :]
        p_center = points[:, 1:-1, :]
        p_right = points[:, 2:, :]
        
        # Second derivative approximation
        second_derivative = p_right - 2 * p_center + p_left
        
        # Mean squared magnitude
        return torch.mean(torch.sum(second_derivative ** 2, dim=2))

    def length_loss(self, points):
        """
        Calculates variance in segment lengths.
        The network should keep points equidistant.
        """
        points = points.float()
        
        # Calculate vector between consecutive points
        diffs = points[:, 1:, :] - points[:, :-1, :] # (B, N-1, 3)
        
        # Calculate lengths (Euclidean distance)
        segment_lengths = torch.norm(diffs, dim=2) # (B, N-1)
        
        # We want all segments to be equal length, so minimize variance
        mean_len = torch.mean(segment_lengths, dim=1, keepdim=True)
        variance = torch.mean((segment_lengths - mean_len) ** 2)
        
        return variance

    def project_3d_to_2d(self, points_3d, proj_mats):
        """
        Differentiable 3D->2D projection.
        points_3d: (B, N, 3)
        proj_mats: (B, 4, 4)
        """
        # --- CRITICAL FIX: Force everything to float32 ---
        points_3d = points_3d.float()
        proj_mats = proj_mats.float()
        
        B, N, _ = points_3d.shape
        
        # Add homogeneous coordinate (ones) -> (B, N, 4)
        ones = torch.ones(B, N, 1, device=points_3d.device, dtype=torch.float32)
        points_h = torch.cat([points_3d, ones], dim=2)
        
        # Batch Matrix Multiplication: (B, 4, 4) x (B, N, 4)^T
        # Reshape points to (B, 4, N) for multiplication
        points_h_t = points_h.permute(0, 2, 1) # (B, 4, N)
        
        # Perform projection
        projected = torch.bmm(proj_mats, points_h_t) # (B, 4, N)
        
        # Convert back to (B, N, 4)
        projected = projected.permute(0, 2, 1)
        
        # Extract u, v
        uv = projected[:, :, :2]
        
        return uv

    def forward(self, pred_3d, target_3d, proj_mats):
        """
        pred_3d: Network output (B, N, 3)
        target_3d: Ground truth centerline (B, N, 3)
        proj_mats: Camera matrices (B, 4, 4)
        """
        losses = {}
        
        # Ensure target is float32
        target_3d = target_3d.float()
        
        # 1. Supervised Loss (Match Ground Truth)
        losses['mse'] = self.mse(pred_3d, target_3d) * self.w_mse
        
        # 2. Physics Loss (Internal Energy)
        losses['curvature'] = self.curvature_loss(pred_3d) * self.w_curvature
        losses['length'] = self.length_loss(pred_3d) * self.w_length
        
        # 3. Projection Loss (Visual Consistency)
        pred_2d = self.project_3d_to_2d(pred_3d, proj_mats)
        target_2d = self.project_3d_to_2d(target_3d, proj_mats)
        
        losses['proj'] = self.mse(pred_2d, target_2d) * self.w_proj
        
        # Total
        total_loss = sum(losses.values())
        
        return total_loss, losses