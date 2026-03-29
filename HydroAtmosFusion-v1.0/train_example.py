#!/usr/bin/env python3
"""Example training script for HydroAtmosFusion.

This script demonstrates how to train the HydroAtmosFusion model on
ocean-atmosphere data for extreme rainfall prediction.

Usage:
    python train_example.py --data_path data.npz --epochs 20 --batch_size 32
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from tqdm import tqdm
import sys

from hydroatmosfusion.models import HydroAtmosFusion
from hydroatmosfusion.losses import CompositeLoss
from hydroatmosfusion.metrics import evaluate_all
from hydroatmosfusion.dataset import create_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train HydroAtmosFusion for extreme rainfall prediction'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to .npz data file',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory to save outputs',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Initial learning rate',
    )
    parser.add_argument(
        '--temporal_steps',
        type=int,
        default=4,
        help='Number of temporal steps',
    )
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=256,
        help='Feature dimension',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=5,
        help='Save model every N epochs',
    )
    parser.add_argument(
        '--use_tensorboard',
        action='store_true',
        help='Use TensorBoard for logging',
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def main():
    """Main training function."""
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}")
    print(f"Output directory: {output_dir}")

    # Create dataloaders
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            temporal_steps=args.temporal_steps,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = HydroAtmosFusion(
        ocean_channels=3,
        atmo_channels=3,
        temporal_steps=args.temporal_steps,
        num_climate_indices=3,
        feature_dim=args.feature_dim,
        num_attention_heads=8,
        attention_d_per_head=32,
        use_climate_indices=True,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Loss and optimizer
    loss_fn = CompositeLoss(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
        wmse_lambda=5.0,
        wmse_percentile=99.0,
        fss_radius=5,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # TensorBoard
    writer = None
    if args.use_tensorboard:
        writer = SummaryWriter(output_dir / 'logs')

    # Training history
    history = {
        'train_loss': [],
        'val_csi': [],
        'val_rmse': [],
        'val_fss': [],
    }

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")

    best_val_csi = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_components = {'wmse': 0, 'focal_csi': 0, 'fss': 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            ocean = batch['ocean'].to(device)
            atmosphere = batch['atmosphere'].to(device)
            climate = batch['climate_indices'].to(device)
            rainfall = batch['rainfall'].unsqueeze(1).to(device)

            # Forward pass
            pred = model(ocean, atmosphere, climate)
            loss, components = loss_fn(pred, rainfall, return_components=True)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            train_loss += loss.item()
            for key in train_loss_components:
                train_loss_components[key] += components[key]

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average training loss
        num_batches = len(train_loader)
        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_metrics_list = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in pbar:
                ocean = batch['ocean'].to(device)
                atmosphere = batch['atmosphere'].to(device)
                climate = batch['climate_indices'].to(device)
                rainfall = batch['rainfall'].unsqueeze(1).to(device)

                pred = model(ocean, atmosphere, climate)
                metrics = evaluate_all(pred, rainfall, percentile=99.0)
                val_metrics_list.append(metrics)

                pbar.set_postfix({
                    'csi': f'{metrics["csi"]:.4f}',
                    'rmse': f'{metrics["rmse"]:.4f}',
                })

        # Average validation metrics
        avg_val_csi = sum(m['csi'] for m in val_metrics_list) / len(val_metrics_list)
        avg_val_rmse = sum(m['rmse'] for m in val_metrics_list) / len(val_metrics_list)
        avg_val_fss = sum(m['fss'] for m in val_metrics_list) / len(val_metrics_list)

        history['val_csi'].append(avg_val_csi)
        history['val_rmse'].append(avg_val_rmse)
        history['val_fss'].append(avg_val_fss)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"    - WMSE: {train_loss_components['wmse']/num_batches:.4f}")
        print(f"    - Focal CSI: {train_loss_components['focal_csi']/num_batches:.4f}")
        print(f"    - FSS: {train_loss_components['fss']/num_batches:.4f}")
        print(f"  Val CSI: {avg_val_csi:.4f}")
        print(f"  Val RMSE: {avg_val_rmse:.4f}")
        print(f"  Val FSS: {avg_val_fss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Metrics/val_csi', avg_val_csi, epoch)
            writer.add_scalar('Metrics/val_rmse', avg_val_rmse, epoch)
            writer.add_scalar('Metrics/val_fss', avg_val_fss, epoch)

        # Save best model
        if avg_val_csi > best_val_csi:
            best_val_csi = avg_val_csi
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                output_dir / 'best_model.pt',
            )
            print(f"  -> Saved best model (CSI: {best_val_csi:.4f})")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                },
                output_dir / f'checkpoint_epoch_{epoch+1}.pt',
            )
            print(f"  -> Saved checkpoint")

        # Learning rate step
        scheduler.step()

    # Testing phase
    print(f"\n{'='*60}")
    print("Testing on test set...")
    print(f"{'='*60}\n")

    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    model.eval()

    test_metrics_list = []
    pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch in pbar:
            ocean = batch['ocean'].to(device)
            atmosphere = batch['atmosphere'].to(device)
            climate = batch['climate_indices'].to(device)
            rainfall = batch['rainfall'].unsqueeze(1).to(device)

            pred = model(ocean, atmosphere, climate)
            metrics = evaluate_all(pred, rainfall, percentile=99.0)
            test_metrics_list.append(metrics)

    # Average test metrics
    test_results = {
        k: sum(m[k] for m in test_metrics_list) / len(test_metrics_list)
        for k in test_metrics_list[0].keys()
    }

    print("Test Results:")
    for name, value in test_results.items():
        if name != 'threshold':
            print(f"  {name}: {value:.4f}")

    # Save results
    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_val_csi': best_val_csi,
        'test_results': test_results,
        'history': history,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if writer is not None:
        writer.close()

    print(f"\nTraining complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
