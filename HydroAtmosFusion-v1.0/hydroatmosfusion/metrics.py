"""Evaluation metrics for HydroAtmosFusion framework.

Implements comprehensive evaluation metrics for rainfall prediction:
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- Correlation: Spatial correlation coefficient
- CSI: Critical Success Index
- EDI: Extremal Dependence Index
- FSS: Fractions Skill Score
- evaluate_all: Comprehensive evaluation function
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


def compute_rmse(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> float:
    """Compute Root Mean Squared Error.

    RMSE = sqrt(mean((pred - true)^2))

    Args:
        pred: Predicted rainfall of shape (B, 1, H, W) or (B, H, W)
        true: Target rainfall of shape (B, 1, H, W) or (B, H, W)

    Returns:
        RMSE value as float
    """
    # Squeeze singleton dimension if present
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if true.dim() == 4 and true.shape[1] == 1:
        true = true.squeeze(1)

    mse = torch.mean((pred - true) ** 2)
    rmse = torch.sqrt(mse + 1e-7)

    return rmse.item()


def compute_mae(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> float:
    """Compute Mean Absolute Error.

    MAE = mean(|pred - true|)

    Args:
        pred: Predicted rainfall of shape (B, 1, H, W) or (B, H, W)
        true: Target rainfall of shape (B, 1, H, W) or (B, H, W)

    Returns:
        MAE value as float
    """
    # Squeeze singleton dimension if present
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if true.dim() == 4 and true.shape[1] == 1:
        true = true.squeeze(1)

    mae = torch.mean(torch.abs(pred - true))

    return mae.item()


def compute_correlation(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> float:
    """Compute spatial correlation coefficient.

    Computes Pearson correlation coefficient between prediction and target,
    treating spatial-temporal domain as flattened samples.

    Args:
        pred: Predicted rainfall of shape (B, 1, H, W) or (B, H, W)
        true: Target rainfall of shape (B, 1, H, W) or (B, H, W)

    Returns:
        Correlation coefficient in [-1, 1]
    """
    # Squeeze singleton dimension if present
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if true.dim() == 4 and true.shape[1] == 1:
        true = true.squeeze(1)

    # Flatten to 1D
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # Compute means
    pred_mean = torch.mean(pred_flat)
    true_mean = torch.mean(true_flat)

    # Compute correlation
    numerator = torch.sum((pred_flat - pred_mean) * (true_flat - true_mean))
    denominator = torch.sqrt(
        torch.sum((pred_flat - pred_mean) ** 2) *
        torch.sum((true_flat - true_mean) ** 2)
    )

    correlation = numerator / (denominator + 1e-7)

    return correlation.item()


def compute_csi(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: Optional[float] = None,
    percentile: float = 99.0,
) -> float:
    """Compute Critical Success Index (CSI).

    CSI = TP / (TP + FP + FN)

    Measures the fraction of rainfall events correctly predicted,
    emphasizing correct detection of extreme rainfall.

    Args:
        pred: Predicted rainfall of shape (B, 1, H, W) or (B, H, W)
        true: Target rainfall of shape (B, 1, H, W) or (B, H, W)
        threshold (float): Rainfall threshold for event detection.
            If None, uses percentile from true values.
        percentile (float): Percentile for automatic threshold. Default: 99

    Returns:
        CSI value in [0, 1]
    """
    # Squeeze singleton dimension if present
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if true.dim() == 4 and true.shape[1] == 1:
        true = true.squeeze(1)

    # Determine threshold
    if threshold is None:
        threshold = torch.quantile(true, percentile / 100.0)

    # Convert to binary (0 or 1)
    pred_binary = (pred > threshold).float()
    true_binary = (true > threshold).float()

    # Compute confusion matrix
    tp = torch.sum(pred_binary * true_binary)
    fp = torch.sum(pred_binary * (1 - true_binary))
    fn = torch.sum((1 - pred_binary) * true_binary)

    # Compute CSI
    denominator = tp + fp + fn
    csi = tp / (denominator + 1e-7)

    return csi.item()


def compute_edi(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: Optional[float] = None,
    percentile: float = 99.0,
) -> float:
    """Compute Extremal Dependence Index (EDI).

    EDI measures the agreement between predictions and observations
    in the upper tail of the distribution.

    EDI = P(pred > threshold | true > threshold) +
          P(true > threshold | pred > threshold) - 1

    Args:
        pred: Predicted rainfall of shape (B, 1, H, W) or (B, H, W)
        true: Target rainfall of shape (B, 1, H, W) or (B, H, W)
        threshold (float): Rainfall threshold for extreme definition.
            If None, uses percentile from true values.
        percentile (float): Percentile for automatic threshold. Default: 99

    Returns:
        EDI value in [0, 1], where 1 indicates perfect extremal dependence
    """
    # Squeeze singleton dimension if present
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if true.dim() == 4 and true.shape[1] == 1:
        true = true.squeeze(1)

    # Determine threshold
    if threshold is None:
        threshold = torch.quantile(true, percentile / 100.0)

    # Convert to binary
    pred_extreme = (pred > threshold).float()
    true_extreme = (true > threshold).float()

    # Compute conditional probabilities
    # P(pred > t | true > t) = P(pred > t AND true > t) / P(true > t)
    joint = torch.sum(pred_extreme * true_extreme)
    true_count = torch.sum(true_extreme) + 1e-7
    pred_given_true = joint / true_count

    # P(true > t | pred > t) = P(pred > t AND true > t) / P(pred > t)
    pred_count = torch.sum(pred_extreme) + 1e-7
    true_given_pred = joint / pred_count

    # EDI = P(pred|true) + P(true|pred) - 1
    edi = (pred_given_true + true_given_pred - 1.0).clamp(min=0.0)

    return edi.item()


def compute_fss(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: Optional[float] = None,
    percentile: float = 99.0,
    radius: int = 5,
) -> float:
    """Compute Fractions Skill Score (FSS).

    FSS = 1 - (MSE_neighborhood / MSE_perfect)

    Measures spatial pattern skill by comparing neighborhood fractions
    across multiple scales. Values closer to 1 indicate better skill.

    Args:
        pred: Predicted rainfall of shape (B, 1, H, W) or (B, H, W)
        true: Target rainfall of shape (B, 1, H, W) or (B, H, W)
        threshold (float): Rainfall threshold for binary conversion.
            If None, uses percentile from true values.
        percentile (float): Percentile for automatic threshold. Default: 99
        radius (int): Search radius for neighborhood. Default: 5

    Returns:
        FSS value in [0, 1], where 1 is perfect skill
    """
    # Squeeze singleton dimension if present
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if true.dim() == 4 and true.shape[1] == 1:
        true = true.squeeze(1)

    # Determine threshold
    if threshold is None:
        threshold = torch.quantile(true, percentile / 100.0)

    # Convert to binary
    pred_binary = (pred > threshold).float()
    true_binary = (true > threshold).float()

    # Create kernel for neighborhood fraction computation
    kernel = torch.ones(
        1, 1, 2 * radius + 1, 2 * radius + 1,
        dtype=pred.dtype,
        device=pred.device,
    )
    kernel = kernel / kernel.sum()

    # Pad predictions and targets
    pred_padded = F.pad(pred_binary.unsqueeze(1), (radius, radius, radius, radius), mode='reflect')
    true_padded = F.pad(true_binary.unsqueeze(1), (radius, radius, radius, radius), mode='reflect')

    # Compute neighborhood fractions via convolution
    pred_frac = F.conv2d(pred_padded, kernel, padding=0)
    true_frac = F.conv2d(true_padded, kernel, padding=0)

    # Squeeze and compute MSE between fractions
    pred_frac = pred_frac.squeeze(1)
    true_frac = true_frac.squeeze(1)

    mse_neighborhood = torch.mean((pred_frac - true_frac) ** 2)

    # Compute perfect MSE (worst case scenario)
    event_fraction = torch.mean(true_binary)
    mse_perfect = event_fraction * (1.0 - event_fraction)

    # Compute FSS
    fss = 1.0 - (mse_neighborhood / (mse_perfect + 1e-7))
    fss = torch.clamp(fss, min=0.0, max=1.0)

    return fss.item()


def evaluate_all(
    pred: torch.Tensor,
    true: torch.Tensor,
    percentile: float = 99.0,
    threshold: Optional[float] = None,
    fss_radius: int = 5,
) -> Dict[str, float]:
    """Comprehensive evaluation of all metrics.

    Computes all evaluation metrics in a single function call and returns
    them as a dictionary.

    Args:
        pred: Predicted rainfall of shape (B, 1, H, W) or (B, H, W)
        true: Target rainfall of shape (B, 1, H, W) or (B, H, W)
        percentile (float): Percentile for extreme rainfall definition. Default: 99
        threshold (float): Explicit rainfall threshold. If None, derived from percentile.
        fss_radius (int): Search radius for FSS computation. Default: 5

    Returns:
        Dictionary with metrics:
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'correlation': Spatial correlation coefficient
            - 'csi': Critical Success Index
            - 'edi': Extremal Dependence Index
            - 'fss': Fractions Skill Score
            - 'threshold': Threshold used for binary classification
    """
    # Determine threshold
    if threshold is None:
        if pred.dim() == 4 and pred.shape[1] == 1:
            threshold_val = torch.quantile(true.squeeze(1), percentile / 100.0)
        else:
            threshold_val = torch.quantile(true, percentile / 100.0)
    else:
        threshold_val = threshold

    # Compute all metrics
    metrics = {
        'rmse': compute_rmse(pred, true),
        'mae': compute_mae(pred, true),
        'correlation': compute_correlation(pred, true),
        'csi': compute_csi(pred, true, threshold=threshold_val.item(), percentile=percentile),
        'edi': compute_edi(pred, true, threshold=threshold_val.item(), percentile=percentile),
        'fss': compute_fss(pred, true, threshold=threshold_val.item(), percentile=percentile, radius=fss_radius),
        'threshold': threshold_val.item(),
    }

    return metrics


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)

    B, H, W = 4, 64, 64

    # Create dummy data with some correlation
    noise_pred = torch.randn(B, 1, H, W) * 10
    base = torch.randn(H, W) * 30
    true = base.unsqueeze(0).repeat(B, 1, 1).unsqueeze(1) + noise_pred
    true = torch.abs(true)  # Make positive

    pred = true + torch.randn_like(true) * 5

    print("Computing individual metrics...")
    print(f"RMSE: {compute_rmse(pred, true):.4f} mm")
    print(f"MAE: {compute_mae(pred, true):.4f} mm")
    print(f"Correlation: {compute_correlation(pred, true):.4f}")
    print(f"CSI (P99): {compute_csi(pred, true, percentile=99.0):.4f}")
    print(f"EDI (P99): {compute_edi(pred, true, percentile=99.0):.4f}")
    print(f"FSS (P99): {compute_fss(pred, true, percentile=99.0, radius=5):.4f}")

    print("\nComputing all metrics together...")
    all_metrics = evaluate_all(pred, true, percentile=99.0, fss_radius=5)
    for metric_name, metric_value in all_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
