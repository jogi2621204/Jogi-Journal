"""
Training script for HydroAtmosFusion model.

Trains the model on synthetic oceanographic and atmospheric data with various
metrics tracking and model checkpointing capabilities.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and prepare data loaders.

    Args:
        data_path: Path to .npz file with synthetic data
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Loading data from {data_path}")

    data = np.load(data_path, allow_pickle=True)

    # Extract ocean and atmospheric variables
    sst = data['sst']  # (num_days, grid_size, grid_size)
    ssh = data['ssh']
    eke = data['eke']
    cape = data['cape']
    tcwv = data['tcwv']
    wind_speed = data['wind_speed']

    # Get climate indices
    oni = data['oni']
    dmi = data['dmi']
    rmm1 = data['rmm1']
    rmm2 = data['rmm2']

    rainfall = data['rainfall']

    num_days = sst.shape[0]
    grid_size = config['data']['grid_size']

    logger.info(f"Loaded data: {num_days} days, {grid_size}x{grid_size} grid")

    # Prepare inputs: [SST, SSH, EKE, CAPE, TCWV, wind_speed]
    ocean_vars = np.stack([sst, ssh, eke], axis=-1)  # (T, H, W, 3)
    atmo_vars = np.stack([cape, tcwv, wind_speed], axis=-1)  # (T, H, W, 3)

    # Climate indices as additional features
    climate_indices = np.stack([oni, dmi, rmm1, rmm2], axis=-1)  # (T, 4)
    # Expand to spatial dimensions
    climate_indices_spatial = np.tile(
        climate_indices[:, np.newaxis, np.newaxis, :],
        (1, grid_size, grid_size, 1)
    )  # (T, H, W, 4)

    # Combine all inputs
    X = np.concatenate([ocean_vars, atmo_vars, climate_indices_spatial], axis=-1)
    # X shape: (T, H, W, 10) = [SST, SSH, EKE, CAPE, TCWV, wind_speed, ONI, DMI, RMM1, RMM2]
    y = rainfall  # (T, H, W)

    # Normalize inputs and outputs
    X_mean = X.mean(axis=(0, 1, 2), keepdims=True)
    X_std = X.std(axis=(0, 1, 2), keepdims=True)
    X_std = np.where(X_std == 0, 1, X_std)  # Avoid division by zero
    X = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std

    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")

    # Split data into train/val/test
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']

    train_end = int(num_days * train_ratio)
    val_end = train_end + int(num_days * val_ratio)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, {'X_mean': X_mean, 'X_std': X_std,
                                                     'y_mean': y_mean, 'y_std': y_std}


class SimpleCNN(nn.Module):
    """Simple CNN model for rainfall prediction."""

    def __init__(self, input_channels: int = 10):
        """
        Initialize CNN model.

        Args:
            input_channels: Number of input channels
        """
        super().__init__()

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
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, 1, H, W)
        """
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        feat = self.features(x)
        out = self.decoder(feat)
        return out.squeeze(1)  # (B, H, W)


class RainfallLoss(nn.Module):
    """Weighted loss function for rainfall prediction."""

    def __init__(self, alpha: float = 0.4, beta: float = 0.35, gamma: float = 0.25,
                 lambda_extreme: float = 5.0):
        """
        Initialize loss function.

        Args:
            alpha: Weight for MSE loss
            beta: Weight for focal loss
            gamma: Weight for FSS loss
            lambda_extreme: Weight for extreme events
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_extreme = lambda_extreme

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss.

        Args:
            pred: Predicted rainfall (B, H, W)
            target: Target rainfall (B, H, W)

        Returns:
            Loss value
        """
        # MSE loss
        mse_loss = nn.functional.mse_loss(pred, target)

        # Focal loss for prediction focus
        focal_weight = torch.abs(pred - target) ** 2
        focal_loss = (focal_weight * mse_loss).mean()

        # Extreme event weight
        target_std = target.std()
        extreme_threshold = target.mean() + 2 * target_std
        extreme_mask = target > extreme_threshold
        extreme_weight = torch.ones_like(target)
        extreme_weight[extreme_mask] = self.lambda_extreme

        weighted_mse = (extreme_weight * (pred - target) ** 2).mean()

        # Total loss
        total_loss = (self.alpha * weighted_mse +
                     self.beta * focal_loss +
                     self.gamma * mse_loss)

        return total_loss


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        pred: Predicted rainfall
        target: Target rainfall

    Returns:
        Dictionary of metrics
    """
    # RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    # MAE
    mae = np.mean(np.abs(pred - target))

    # Correlation
    correlation = np.corrcoef(pred.flatten(), target.flatten())[0, 1]

    # CSI99: Critical Success Index for P99
    p99_pred = np.percentile(pred, 99)
    p99_target = np.percentile(target, 99)

    pred_extreme = pred > p99_pred
    target_extreme = target > p99_target

    hits = np.sum(pred_extreme & target_extreme)
    false_alarms = np.sum(pred_extreme & ~target_extreme)
    misses = np.sum(~pred_extreme & target_extreme)

    csi99 = hits / (hits + false_alarms + misses + 1e-8)

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation),
        'csi99': float(csi99),
    }


def train_epoch(model: nn.Module, train_loader: DataLoader,
                optimizer: torch.optim.Optimizer, loss_fn: RainfallLoss,
                device: torch.device) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on

    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


def validate(model: nn.Module, val_loader: DataLoader,
            loss_fn: RainfallLoss, device: torch.device) -> Tuple[float, Dict]:
    """
    Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on

    Returns:
        Tuple of (loss, metrics)
    """
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_target = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            all_pred.append(pred.cpu().numpy())
            all_target.append(y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_target = np.concatenate(all_target)
    metrics = compute_metrics(all_pred, all_target)

    return total_loss / len(val_loader), metrics


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train HydroAtmosFusion model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--data_path', type=str, default='data/synthetic_ocean_atmos.npz',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                       else 'cpu', help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Override config with arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    logger.info(f"Training configuration: {config}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader, normalization = load_data(
        args.data_path, config
    )

    # Create model
    device = torch.device(args.device)
    model = SimpleCNN(input_channels=10).to(device)
    logger.info(f"Model initialized on {device}")

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )

    # Create loss function
    loss_fn = RainfallLoss(
        alpha=config['loss']['alpha'],
        beta=config['loss']['beta'],
        gamma=config['loss']['gamma'],
        lambda_extreme=config['loss']['lambda_extreme']
    ).to(device)

    # TensorBoard writer
    tb_dir = output_dir / 'logs'
    tb_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    logger.info("=" * 70)
    logger.info("Starting training")
    logger.info("=" * 70)

    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        logger.info(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss, metrics = validate(model, val_loader, loss_fn, device)
        logger.info(f"Val Loss: {val_loss:.6f}")
        logger.info(f"Val Metrics - RMSE: {metrics['rmse']:.4f}, "
                   f"MAE: {metrics['mae']:.4f}, "
                   f"Corr: {metrics['correlation']:.4f}, "
                   f"CSI99: {metrics['csi99']:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/rmse', metrics['rmse'], epoch)
        writer.add_scalar('Metrics/csi99', metrics['csi99'], epoch)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
                'config': config,
                'normalization': normalization,
            }, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")

        scheduler.step()

    logger.info("=" * 70)
    logger.info(f"Training complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")
    logger.info("=" * 70)

    writer.close()


if __name__ == '__main__':
    main()
