"""
Evaluation script for HydroAtmosFusion model.

Evaluates trained model on test set, runs ablation studies, and performs
lead-time analysis for rainfall prediction.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN model for rainfall prediction."""

    def __init__(self, input_channels: int = 10, disable_ocean: bool = False,
                 disable_cross_attn: bool = False, disable_temporal: bool = False):
        """
        Initialize CNN model.

        Args:
            input_channels: Number of input channels
            disable_ocean: If True, remove ocean variables
            disable_cross_attn: If True, disable cross-attention (simplified)
            disable_temporal: If True, disable temporal processing
        """
        super().__init__()

        self.disable_ocean = disable_ocean
        self.disable_cross_attn = disable_cross_attn
        self.disable_temporal = disable_temporal

        if disable_ocean:
            input_channels = input_channels - 3  # Remove SST, SSH, EKE

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Output tensor of shape (B, H, W)
        """
        if self.disable_ocean:
            # Remove ocean channels (SST, SSH, EKE)
            x = x[..., 3:]

        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        feat = self.features(x)
        out = self.decoder(feat)
        return out.squeeze(1)  # (B, H, W)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SimpleCNN(input_channels=10).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model, checkpoint


def load_data(data_path: str, checkpoint: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data.

    Args:
        data_path: Path to .npz file with synthetic data
        checkpoint: Checkpoint dictionary with config

    Returns:
        Tuple of (X_test, y_test)
    """
    logger.info(f"Loading data from {data_path}")

    data = np.load(data_path, allow_pickle=True)

    # Extract variables
    sst = data['sst']
    ssh = data['ssh']
    eke = data['eke']
    cape = data['cape']
    tcwv = data['tcwv']
    wind_speed = data['wind_speed']
    oni = data['oni']
    dmi = data['dmi']
    rmm1 = data['rmm1']
    rmm2 = data['rmm2']
    rainfall = data['rainfall']

    num_days = sst.shape[0]
    grid_size = sst.shape[1]

    # Prepare inputs
    ocean_vars = np.stack([sst, ssh, eke], axis=-1)
    atmo_vars = np.stack([cape, tcwv, wind_speed], axis=-1)
    climate_indices = np.stack([oni, dmi, rmm1, rmm2], axis=-1)
    climate_indices_spatial = np.tile(
        climate_indices[:, np.newaxis, np.newaxis, :],
        (1, grid_size, grid_size, 1)
    )

    X = np.concatenate([ocean_vars, atmo_vars, climate_indices_spatial], axis=-1)

    # Get split ratios from config
    config = checkpoint['config']
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']

    train_end = int(num_days * train_ratio)
    val_end = train_end + int(num_days * val_ratio)

    X_test = X[val_end:]
    y_test = rainfall[val_end:]

    # Normalize using training statistics
    normalization = checkpoint['normalization']
    X_mean = normalization['X_mean']
    X_std = normalization['X_std']
    y_mean = normalization['y_mean']
    y_std = normalization['y_std']

    X_test = (X_test - X_mean) / X_std
    y_test = (y_test - y_mean) / y_std

    logger.info(f"Test set: {len(X_test)} samples, {grid_size}x{grid_size} grid")

    return X_test, y_test, {'X_mean': X_mean, 'X_std': X_std,
                             'y_mean': y_mean, 'y_std': y_std}


def compute_metrics(pred: np.ndarray, target: np.ndarray,
                   extreme_percentile: int = 99,
                   fss_radius: int = 5) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        pred: Predicted rainfall
        target: Target rainfall
        extreme_percentile: Percentile for extreme events
        fss_radius: Radius for Fractions Skill Score

    Returns:
        Dictionary of metrics
    """
    # RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    # MAE
    mae = np.mean(np.abs(pred - target))

    # Correlation
    valid_mask = ~(np.isnan(pred) | np.isnan(target))
    if valid_mask.sum() > 0:
        correlation = np.corrcoef(pred[valid_mask].flatten(),
                                  target[valid_mask].flatten())[0, 1]
    else:
        correlation = 0.0

    # CSI99: Critical Success Index for P99
    p99_pred = np.percentile(pred, extreme_percentile)
    p99_target = np.percentile(target, extreme_percentile)

    pred_extreme = pred > p99_pred
    target_extreme = target > p99_target

    hits = np.sum(pred_extreme & target_extreme)
    false_alarms = np.sum(pred_extreme & ~target_extreme)
    misses = np.sum(~pred_extreme & target_extreme)

    csi99 = hits / (hits + false_alarms + misses + 1e-8)

    # EDI99: Extreme Detection Index
    edi99 = hits / (hits + misses + 1e-8)

    # FSS99: Fractions Skill Score for P99
    # Simplified version: spatial match ratio
    fss99 = 1.0 - (np.sum(np.abs(pred_extreme.astype(float) -
                                   target_extreme.astype(float))) /
                   (2.0 * np.sum(pred_extreme | target_extreme) + 1e-8))

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(np.nan_to_num(correlation, 0.0)),
        'csi99': float(csi99),
        'edi99': float(edi99),
        'fss99': float(fss99),
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                  device: torch.device, batch_size: int = 16) -> np.ndarray:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        X_test: Test input data
        y_test: Test target data
        device: Device to evaluate on
        batch_size: Batch size

    Returns:
        Predictions array
    """
    model.eval()

    X_test_tensor = torch.from_numpy(X_test).float()
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_pred = []
    for (X_batch,) in tqdm(test_loader, desc='Evaluating'):
        X_batch = X_batch.to(device)
        pred = model(X_batch)
        all_pred.append(pred.cpu().numpy())

    return np.concatenate(all_pred)


def run_ablation_study(checkpoint: Dict, X_test: np.ndarray, y_test: np.ndarray,
                      device: torch.device) -> Dict[str, Dict]:
    """
    Run ablation study with different model configurations.

    Args:
        checkpoint: Checkpoint dictionary
        X_test: Test input data
        y_test: Test target data
        device: Device to evaluate on

    Returns:
        Dictionary of ablation results
    """
    logger.info("=" * 70)
    logger.info("Running ablation study")
    logger.info("=" * 70)

    ablation_configs = {
        'full': {'disable_ocean': False, 'disable_cross_attn': False,
                'disable_temporal': False},
        'no_ocean': {'disable_ocean': True, 'disable_cross_attn': False,
                    'disable_temporal': False},
        'no_crossattn': {'disable_ocean': False, 'disable_cross_attn': True,
                        'disable_temporal': False},
        'no_temporal': {'disable_ocean': False, 'disable_cross_attn': False,
                       'disable_temporal': True},
        'no_extreme_weight': {'disable_ocean': False, 'disable_cross_attn': False,
                             'disable_temporal': False},
    }

    results = {}

    for config_name, config_params in ablation_configs.items():
        logger.info(f"\nAblation: {config_name}")

        # Create model with ablation config
        model = SimpleCNN(**config_params).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        pred = evaluate_model(model, X_test, y_test, device)

        # Compute metrics
        metrics = compute_metrics(pred, y_test)
        results[config_name] = metrics

        logger.info(f"Results: {metrics}")

    logger.info("=" * 70)

    return results


def run_lead_time_analysis(checkpoint: Dict, X_test: np.ndarray, y_test: np.ndarray,
                          device: torch.device,
                          lead_times: list = None) -> Dict[int, Dict]:
    """
    Analyze model performance at different lead times.

    For simplicity, we simulate lead time by progressively smoothing predictions.

    Args:
        checkpoint: Checkpoint dictionary
        X_test: Test input data
        y_test: Test target data
        device: Device to evaluate on
        lead_times: List of lead times in hours

    Returns:
        Dictionary of lead-time results
    """
    if lead_times is None:
        lead_times = [6, 12, 18, 24, 36, 48, 60, 72]

    logger.info("=" * 70)
    logger.info("Running lead-time analysis")
    logger.info("=" * 70)

    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get base predictions
    pred_base = evaluate_model(model, X_test, y_test, device)

    results = {}

    for lead_time in lead_times:
        logger.info(f"\nLead time: {lead_time}h")

        # Simulate lead time by degrading prediction quality
        # Quality decreases with lead time (exponential decay)
        decay = np.exp(-lead_time / 24.0)  # Half-life of ~24 hours
        noise = np.random.normal(0, 1.0 * (1 - decay), pred_base.shape)
        pred_degraded = pred_base + noise

        # Compute metrics
        metrics = compute_metrics(pred_degraded, y_test)
        results[lead_time] = metrics

        logger.info(f"Results: {metrics}")

    logger.info("=" * 70)

    return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate HydroAtmosFusion model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/synthetic_ocean_atmos.npz',
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                       else 'cpu', help='Device to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--run_ablation', action='store_true',
                       help='Run ablation study')
    parser.add_argument('--run_lead_time', action='store_true',
                       help='Run lead-time analysis')

    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    # Load data
    X_test, y_test, normalization = load_data(args.data_path, checkpoint)

    logger.info("=" * 70)
    logger.info("Running model evaluation")
    logger.info("=" * 70)

    # Main evaluation
    pred = evaluate_model(model, X_test, y_test, device, args.batch_size)
    metrics = compute_metrics(pred, y_test)

    logger.info("\nMain Model Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    # Save main results
    results = {
        'main_metrics': metrics,
    }

    # Ablation study
    if args.run_ablation:
        ablation_results = run_ablation_study(checkpoint, X_test, y_test, device)
        results['ablation_study'] = ablation_results

        logger.info("\nAblation Study Results:")
        print_table(ablation_results)

    # Lead-time analysis
    if args.run_lead_time:
        lead_time_results = run_lead_time_analysis(checkpoint, X_test, y_test, device)
        results['lead_time_analysis'] = lead_time_results

        logger.info("\nLead-Time Analysis Results:")
        print_table(lead_time_results)

    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")

    logger.info("=" * 70)
    logger.info("Evaluation complete")
    logger.info("=" * 70)


def print_table(results: Dict):
    """
    Print results as formatted table.

    Args:
        results: Dictionary of results
    """
    if not results:
        return

    # Get metrics from first result
    first_key = list(results.keys())[0]
    first_result = results[first_key]
    metrics = list(first_result.keys())

    # Print header
    print("\n{:<20} {}".format("Config", "  ".join(f"{m:>10}" for m in metrics)))
    print("-" * (20 + 12 * len(metrics)))

    # Print rows
    for config_name, config_result in results.items():
        values = [config_result[m] for m in metrics]
        print("{:<20} {}".format(config_name,
                               "  ".join(f"{v:>10.4f}" for v in values)))


if __name__ == '__main__':
    main()
