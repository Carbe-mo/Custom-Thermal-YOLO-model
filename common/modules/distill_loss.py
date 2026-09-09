"""
common/modules/distill_loss.py — Cross-Modality Feature Distillation Loss
========================================================================

Mathematical Design & Choice Rationale:
---------------------------------------
Why Cosine Similarity over L2 Distance for RGB-to-Thermal Distillation?
1. Physical Domain Mismatch:
   - Visible RGB cameras measure surface reflectance and photometric color (3 visible bands).
   - Thermal IR cameras measure long-wave infrared thermal radiance and emissivity.
   - Consequently, intermediate feature activations in an RGB network inherently have
     different dynamic ranges and magnitude scales compared to a Thermal network.

2. Problem with L2 / MSE Distance:
   - L2 distance, ||F_student - F_teacher||^2 = ||F_S||^2 + ||F_T||^2 - 2<F_S, F_T>,
     strictly penalizes differences in absolute activation magnitudes.
   - Forcing a thermal student to mimic the raw numerical magnitude of an RGB teacher
     distorts the thermal model's native gradient distributions and degrades performance.

3. Advantage of Cosine Feature Alignment (Normalized Directional Distillation):
   - L_distill = 1 - <F_student, F_teacher> / (||F_student||_2 * ||F_teacher||_2 + eps)
   - Cosine similarity normalizes out absolute scale differences and aligns only the
     *directional semantic orientation* and relative spatial activation contours.
   - This transfers rich semantic structural cues (object silhouettes, edge contours,
     fine textures) from the RGB teacher to the thermal student without corrupting
     the thermal sensor's natural response characteristics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineFeatureDistillLoss(nn.Module):
    """
    Multi-Scale Cosine Feature Alignment Loss for Knowledge Distillation.
    Computes normalized cosine distance between intermediate student and teacher feature maps.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat_student: torch.Tensor, feat_teacher: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_student: Tensor of shape [B, C, H, W]
            feat_teacher: Tensor of shape [B, C, H, W] (detached / frozen)
        Returns:
            Scalar cosine distillation loss (lower is better, range [0, 2])
        """
        # Normalize along channel dimension
        s_norm = F.normalize(feat_student, p=2, dim=1, eps=self.eps)
        t_norm = F.normalize(feat_teacher.detach(), p=2, dim=1, eps=self.eps)
        
        # Pointwise cosine similarity across spatial grid, averaged over batch and spatial dims
        cos_sim = torch.sum(s_norm * t_norm, dim=1) # [B, H, W]
        cos_loss = 1.0 - torch.mean(cos_sim)
        return cos_loss


class MultiScaleDistillLoss(nn.Module):
    """
    Multi-Scale Distillation Loss across P3, P4, and P5 feature pyramid levels.
    """
    def __init__(self, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.loss_fn = CosineFeatureDistillLoss()
        self.weights = weights

    def forward(
        self,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            student_feats: [P3, P4, P5] student feature maps
            teacher_feats: [P3, P4, P5] teacher feature maps
        """
        total_loss = 0.0
        for i, (f_s, f_t) in enumerate(zip(student_feats, teacher_feats)):
            total_loss += self.weights[i] * self.loss_fn(f_s, f_t)
        return total_loss / sum(self.weights)


if __name__ == "__main__":
    B, C, H, W = 4, 64, 40, 40
    f_student = torch.randn(B, C, H, W, requires_grad=True)
    f_teacher = torch.randn(B, C, H, W)
    
    loss_fn = CosineFeatureDistillLoss()
    loss = loss_fn(f_student, f_teacher)
    loss.backward()
    
    print(f"Cosine Distill Loss: {loss.item():.4f}")
    print(f"Student Gradient Norm: {f_student.grad.norm().item():.4f}")
    assert loss.item() >= 0.0
    print("Distill loss unit test passed successfully!")
