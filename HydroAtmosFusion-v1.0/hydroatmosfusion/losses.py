"""Loss functions for HydroAtmosFusion framework.

Implements all loss components for extreme rainfall prediction:
- WeightedMSELoss: Emphasis on extreme rainfall events (Eq. 22)
- FocalCSILoss: Critical Success Index with focal loss (Eq. 23)
- FSSLoss: Fractions Skill Score loss (Eq. 24)
- CompositeLoss: Weighted combination of all losses (Eq. 25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss emphasizing extreme rainfall events (Eq. 22).

    Standard MSE loss with adaptive weighting that emphasizes extreme rainfall
    events (above P99 threshold). The weight function is:
        w(y) = 1 + lambda * indicator(y > P99)

    where lambda controls the emphasis on extreme events.

    Args:
        lambda_weight (float): Weight multiplier for extreme events. Default: 5
        percentile (float): Percentile threshold for extreme classification. Default: 99
    """

    def __init__(self, lambda_weight: float = 5.0, percentile: float = 99.0):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.percentile = percentile

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            pred: Predicted rainfall of shape (B, 1, H, W)
            true: Target rainfall of shape (B, 1, H, W)

        Returns:
            Scalar loss value
        """
        # Compute P99 threshold from target values
        threshold = torch.quantile(true, self.percentile / 100.0)

        # Create weights: 1 for normal, (1 + lambda) for extreme
        weights = torch.ones_like(true)
        extreme_mask = true > threshold
        weights[extreme_mask] = 1.0 + self.lambda_weight

        # Weighted MSE
        mse = (pred - true) ** 2
        weighted_mse = mse * weights

        return weighted_mse.mean()


class FocalCSILoss(nn.Module):
    """Focal Critical Success Index Loss (Eq. 23).

    Combines Critical Success Index (CSI) with focal loss term to emphasize
    correct detections while avoiding bias toward frequent non-event predictions.

    CSI = TP / (TP + FP + FN)
    Loss = -(1 - CSI)^gamma * log(CSI + eps)

    Args:
        threshold (float): Rainfall threshold for event detection. Default: None
            If None, uses P99 from data during forward pass.
        gamma (float): Focal loss focusing parameter. Default: 2
        eps (float): Small epsilon for numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        gamma: float = 2.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.threshold = threshold
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal CSI loss.

        Args:
            pred: Predicted rainfall of shape (B, 1, H, W)
            true: Target rainfall of shape (B, 1, H, W)

        Returns:
            Scalar loss value
        """
        # Determine threshold
        if self.threshold is None:
            threshold = torch.quantile(true, 0.99)
        else:
            threshold = self.threshold

        # Convert to binary (0 or 1) at threshold
        pred_binary = (pred > threshold).float()
        true_binary = (true > threshold).float()

        # Compute confusion matrix elements
        tp = (pred_binary * true_binary).sum()
        fp = (pred_binary * (1 - true_binary)).sum()
        fn = ((1 - pred_binary) * true_binary).sum()

        # Compute CSI (Critical Success Index)
        denominator = tp + fp + fn
        csi = tp / (denominator + self.eps)

        # Focal CSI loss: -(1-CSI)^gamma * log(CSI)
        focal_weight = torch.pow(1.0 - csi, self.gamma)
        loss = -focal_weight * torch.log(csi + self.eps)

        return loss


class FSSLoss(nn.Module):
    """Fractions Skill Score Loss (Eq. 24).

    Measures spatial pattern similarity using neighborhood-based comparison.
    FSS evaluates rainfall pattern skill across scales using moving windows.

    FSS = 1 - MSE_neighborhood / MSE_perfect

    Args:
        radius (int): Search radius for neighborhood. Default: 5 pixels
        threshold (float): Rainfall threshold for binary conversion. Default: None
            If None, uses P99 from data during forward pass.
    """

    def __init__(self, radius: int = 5, threshold: Optional[float] = None):
        super().__init__()
        self.radius = radius
        self.threshold = threshold
        self.padding = radius

    def _get_neighborhood_fraction(
        self,
        binary_field: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        """Compute fraction of events in neighborhood.

        Args:
            binary_field: Binary event field (0 or 1)
            radius: Neighborhood radius

        Returns:
            Neighborhood fraction field
        """
        kernel = torch.ones(
            1, 1, 2 * radius + 1, 2 * radius + 1,
            dtype=binary_field.dtype,
            device=binary_field.device,
        )
        kernel = kernel / kernel.sum()

        # Pad for valid convolution
        padded = F.pad(binary_field, (radius, radius, radius, radius), mode='reflect')

        # Compute neighborhood fraction via convolution
        neighborhood = F.conv2d(padded.unsqueeze(1), kernel, padding=0)

        return neighborhood.squeeze(1)

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Fractions Skill Score loss.

        Args:
            pred: Predicted rainfall of shape (B, 1, H, W)
            true: Target rainfall of shape (B, 1, H, W)

        Returns:
            Scalar loss value (1 - FSS)
        """
        # Determine threshold
        if self.threshold is None:
            threshold = torch.quantile(true, 0.99)
        else:
            threshold = self.threshold

        # Convert to binary
        pred_binary = (pred > threshold).float()
        true_binary = (true > threshold).float()

        # Compute neighborhood fractions
        pred_frac = self._get_neighborhood_fraction(pred_binary.squeeze(1), self.radius)
        true_frac = self._get_neighborhood_fraction(true_binary.squeeze(1), self.radius)

        # Compute MSE between fractions
        mse_neighborhood = ((pred_frac - true_frac) ** 2).mean()

        # Compute perfect MSE (worst case)
        n_pixels = pred_binary.numel()
        event_fraction = true_binary.mean()
        mse_perfect = event_fraction * (1.0 - event_fraction)

        # Compute FSS and return 1 - FSS as loss
        fss = 1.0 - (mse_neighborhood / (mse_perfect + 1e-7))
        loss = 1.0 - fss

        return loss


class CompositeLoss(nn.Module):
    """Composite loss combining all components (Eq. 25).

    Combines weighted MSE, focal CSI, and FSS losses with fixed weights:
        L_total = alpha * L_WMSE + beta * L_FocalCSI + gamma * L_FSS

    Args:
        alpha (float): Weight for WeightedMSELoss. Default: 0.4
        beta (float): Weight for FocalCSILoss. Default: 0.35
        gamma (float): Weight for FSSLoss. Default: 0.25
        wmse_lambda (float): Lambda for weighted MSE. Default: 5.0
        wmse_percentile (float): Percentile for weighted MSE. Default: 99.0
        fss_radius (int): Radius for FSS neighborhood. Default: 5
        threshold (float): Rainfall threshold. Default: None (computed from data)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        wmse_lambda: float = 5.0,
        wmse_percentile: float = 99.0,
        fss_radius: int = 5,
        threshold: Optional[float] = None,
    ):
        super().__init__()

        # Validate weights sum to 1
        total_weight = alpha + beta + gamma
        if not np.isclose(total_weight, 1.0, atol=1e-5):
            raise ValueError(
                f"Loss weights must sum to 1.0, got {total_weight}. "
                f"Weights: alpha={alpha}, beta={beta}, gamma={gamma}"
            )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

        # Initialize loss components
        self.wmse_loss = WeightedMSELoss(
            lambda_weight=wmse_lambda,
            percentile=wmse_percentile,
        )

        self.focal_csi_loss = FocalCSILoss(
            threshold=threshold,
            gamma=2.0,
        )

        self.fss_loss = FSSLoss(
            radius=fss_radius,
            threshold=threshold,
        )

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute composite loss.

        Args:
            pred: Predicted rainfall of shape (B, 1, H, W)
            true: Target rainfall of shape (B, 1, H, W)
            return_components (bool): If True, also return individual loss components
                as a dictionary

        Returns:
            Composite loss value, or tuple (loss, components_dict) if
            return_components=True
        """
        # Compute individual losses
        wmse = self.wmse_loss(pred, true)
        focal_csi = self.focal_csi_loss(pred, true)
        fss = self.fss_loss(pred, true)

        # Composite loss
        loss = self.alpha * wmse + self.beta * focal_csi + self.gamma * fss

        if return_components:
            components = {
                'wmse': wmse.item(),
                'focal_csi': focal_csi.item(),
                'fss': fss.item(),
                'composite': loss.item(),
            }
            return loss, components
        else:
            return loss


if __name__ == "__main__":
    # Example usage
    B, H, W = 4, 64, 64

    # Create dummy predictions and targets
    pred = torch.rand(B, 1, H, W) * 100  # Rainfall in mm
    true = torch.rand(B, 1, H, W) * 100

    # Test individual losses
    print("Testing individual losses...")

    wmse = WeightedMSELoss(lambda_weight=5.0, percentile=99.0)
    wmse_loss = wmse(pred, true)
    print(f"Weighted MSE Loss: {wmse_loss.item():.4f}")

    focal_csi = FocalCSILoss(threshold=None, gamma=2.0)
    focal_csi_loss = focal_csi(pred, true)
    print(f"Focal CSI Loss: {focal_csi_loss.item():.4f}")

    fss = FSSLoss(radius=5, threshold=None)
    fss_loss = fss(pred, true)
    print(f"FSS Loss: {fss_loss.item():.4f}")

    # Test composite loss
    print("\nTesting composite loss...")
    composite = CompositeLoss(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
    )

    loss, components = composite(pred, true, return_components=True)
    print(f"Composite Loss: {loss.item():.4f}")
    print(f"Components: {components}")
