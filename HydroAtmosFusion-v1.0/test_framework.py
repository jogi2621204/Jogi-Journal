#!/usr/bin/env python3
"""Comprehensive test script for HydroAtmosFusion framework.

Tests all modules and ensures they work correctly together:
- Models: Architecture and forward passes
- Losses: Loss computations
- Metrics: Evaluation metrics
- Dataset: Data loading and processing
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from hydroatmosfusion import models, losses, metrics, dataset


def test_models():
    """Test all model components."""
    print("=" * 60)
    print("TESTING MODELS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Test parameters
    B, T, H, W = 2, 4, 64, 64

    print("Creating HydroAtmosFusion model...")
    model = models.HydroAtmosFusion(
        ocean_channels=3,
        atmo_channels=3,
        temporal_steps=T,
        num_climate_indices=3,
        feature_dim=256,
        num_attention_heads=8,
        attention_d_per_head=32,
        use_climate_indices=True,
    ).to(device)

    print(f"Model created successfully")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Test forward pass
    print("Testing forward pass...")
    ocean_input = torch.randn(B, 3, H, W).to(device)
    atmo_input = torch.randn(B, T, 3, H, W).to(device)
    climate_input = torch.randn(B, 3).to(device)

    with torch.no_grad():
        rainfall_pred = model(ocean_input, atmo_input, climate_input)

    print(f"Ocean input: {ocean_input.shape}")
    print(f"Atmosphere input: {atmo_input.shape}")
    print(f"Climate indices: {climate_input.shape}")
    print(f"Rainfall output: {rainfall_pred.shape}")
    print(f"Output range: [{rainfall_pred.min():.4f}, {rainfall_pred.max():.4f}]")
    assert rainfall_pred.shape == (B, 1, H, W), f"Unexpected output shape: {rainfall_pred.shape}"
    assert rainfall_pred.min() >= 0 and rainfall_pred.max() <= 1, "Output not in [0, 1]"
    print("✓ Forward pass successful\n")

    # Test individual encoders
    print("Testing individual encoders...")

    ocean_encoder = models.OceanEncoder(num_channels=3, output_channels=256).to(device)
    with torch.no_grad():
        ocean_feat = ocean_encoder(ocean_input)
    print(f"OceanEncoder output: {ocean_feat.shape}")
    assert ocean_feat.shape == (B, 256, H // 8, W // 8), f"Unexpected ocean features shape"
    print("✓ OceanEncoder OK")

    atmo_encoder = models.AtmosphereEncoder(num_channels=3, output_channels=256, temporal_steps=T).to(device)
    with torch.no_grad():
        atmo_feat = atmo_encoder(atmo_input)
    print(f"AtmosphereEncoder output: {atmo_feat.shape}")
    assert atmo_feat.shape == (B, 256, H // 4, W // 4), f"Unexpected atmo features shape"
    print("✓ AtmosphereEncoder OK")

    # Test cross-modal attention
    print("\nTesting cross-modal attention...")
    # Upsample ocean features for attention test
    ocean_feat_upsampled = torch.nn.functional.interpolate(
        ocean_feat, size=(H // 4, W // 4), mode='bilinear', align_corners=False
    )

    cross_attn = models.BidirectionalCrossModalAttention(
        feature_dim=256,
        num_heads=8,
        d_per_head=32,
    ).to(device)

    with torch.no_grad():
        fused = cross_attn(ocean_feat_upsampled, atmo_feat)
    print(f"Fused features shape: {fused.shape}")
    assert fused.shape == (B, 512, H // 4, W // 4), f"Unexpected fused shape"
    print("✓ BidirectionalCrossModalAttention OK")

    # Test climate index encoder
    print("\nTesting climate index encoder...")
    climate_encoder = models.ClimateIndexEncoder(num_indices=3, output_dim=64).to(device)
    with torch.no_grad():
        climate_feat = climate_encoder(climate_input, spatial_shape=(H // 4, W // 4))
    print(f"Climate features shape: {climate_feat.shape}")
    assert climate_feat.shape == (B, 64, H // 4, W // 4), f"Unexpected climate features shape"
    print("✓ ClimateIndexEncoder OK")

    # Test prediction head
    print("\nTesting prediction head...")
    pred_head = models.PredictionHead(input_channels=512 + 64, output_channels=1).to(device)
    with torch.no_grad():
        rainfall = pred_head(torch.cat([fused, climate_feat], dim=1))
    print(f"Rainfall prediction shape: {rainfall.shape}")
    assert rainfall.shape == (B, 1, H // 4, W // 4), f"Unexpected rainfall shape"
    print("✓ PredictionHead OK\n")

    print("=" * 60)
    print("ALL MODEL TESTS PASSED")
    print("=" * 60 + "\n")


def test_losses():
    """Test all loss functions."""
    print("=" * 60)
    print("TESTING LOSSES")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, H, W = 4, 64, 64

    # Create realistic rainfall data
    pred = torch.rand(B, 1, H, W).to(device) * 100  # Rainfall 0-100 mm
    true = torch.rand(B, 1, H, W).to(device) * 100

    print("Testing WeightedMSELoss...")
    wmse = losses.WeightedMSELoss(lambda_weight=5.0, percentile=99.0).to(device)
    wmse_loss = wmse(pred, true)
    print(f"WeightedMSELoss: {wmse_loss.item():.4f}")
    assert wmse_loss.item() > 0, "Loss should be positive"
    print("✓ WeightedMSELoss OK\n")

    print("Testing FocalCSILoss...")
    focal_csi = losses.FocalCSILoss(gamma=2.0).to(device)
    focal_csi_loss = focal_csi(pred, true)
    print(f"FocalCSILoss: {focal_csi_loss.item():.4f}")
    assert focal_csi_loss.item() >= 0, "Loss should be non-negative"
    print("✓ FocalCSILoss OK\n")

    print("Testing FSSLoss...")
    fss = losses.FSSLoss(radius=5).to(device)
    fss_loss = fss(pred, true)
    print(f"FSSLoss: {fss_loss.item():.4f}")
    assert 0 <= fss_loss.item() <= 1, "FSS loss should be in [0, 1]"
    print("✓ FSSLoss OK\n")

    print("Testing CompositeLoss...")
    composite = losses.CompositeLoss(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
    ).to(device)

    loss, components = composite(pred, true, return_components=True)
    print(f"Composite Loss: {loss.item():.4f}")
    print(f"Components:")
    print(f"  - Weighted MSE: {components['wmse']:.4f}")
    print(f"  - Focal CSI: {components['focal_csi']:.4f}")
    print(f"  - FSS: {components['fss']:.4f}")
    assert loss.item() > 0, "Composite loss should be positive"
    print("✓ CompositeLoss OK\n")

    print("=" * 60)
    print("ALL LOSS TESTS PASSED")
    print("=" * 60 + "\n")


def test_metrics():
    """Test all evaluation metrics."""
    print("=" * 60)
    print("TESTING METRICS")
    print("=" * 60)

    B, H, W = 4, 64, 64

    # Create correlated prediction and target
    base = torch.randn(H, W)
    true = base.unsqueeze(0).repeat(B, 1, 1).unsqueeze(1)
    pred = true + torch.randn_like(true) * 5
    pred = torch.abs(pred)  # Ensure positive

    print("Testing individual metrics...")

    rmse = metrics.compute_rmse(pred, true)
    print(f"RMSE: {rmse:.4f}")
    assert rmse > 0, "RMSE should be positive"
    print("✓ RMSE OK")

    mae = metrics.compute_mae(pred, true)
    print(f"MAE: {mae:.4f}")
    assert mae > 0, "MAE should be positive"
    print("✓ MAE OK")

    corr = metrics.compute_correlation(pred, true)
    print(f"Correlation: {corr:.4f}")
    assert -1 <= corr <= 1, "Correlation should be in [-1, 1]"
    print("✓ Correlation OK")

    csi = metrics.compute_csi(pred, true, percentile=99.0)
    print(f"CSI (P99): {csi:.4f}")
    assert 0 <= csi <= 1, "CSI should be in [0, 1]"
    print("✓ CSI OK")

    edi = metrics.compute_edi(pred, true, percentile=99.0)
    print(f"EDI (P99): {edi:.4f}")
    assert 0 <= edi <= 1, "EDI should be in [0, 1]"
    print("✓ EDI OK")

    fss = metrics.compute_fss(pred, true, percentile=99.0, radius=5)
    print(f"FSS (P99): {fss:.4f}")
    assert 0 <= fss <= 1, "FSS should be in [0, 1]"
    print("✓ FSS OK")

    print("\nTesting evaluate_all...")
    all_metrics = metrics.evaluate_all(pred, true, percentile=99.0)
    print("All metrics computed:")
    for name, value in all_metrics.items():
        if name != 'threshold':
            print(f"  - {name}: {value:.4f}")

    print("✓ evaluate_all OK\n")

    print("=" * 60)
    print("ALL METRIC TESTS PASSED")
    print("=" * 60 + "\n")


def test_dataset():
    """Test dataset loading and processing."""
    print("=" * 60)
    print("TESTING DATASET")
    print("=" * 60)

    # Create synthetic data
    print("Creating synthetic dataset...")
    T, H, W = 100, 64, 64

    ocean_data = np.random.randn(T, H, W, 3) * 10 + np.array([25, 0.5, 100])
    atmo_data = np.random.randn(T, H, W, 3) * np.array([200, 10, 5]) + \
                np.array([2000, 50, 5])
    rainfall_data = np.abs(np.random.randn(T, H, W)) * 30 + 10
    climate_data = np.random.randn(T, 3)

    temp_path = '/tmp/test_hydroatmos.npz'
    np.savez(
        temp_path,
        ocean=ocean_data,
        atmosphere=atmo_data,
        rainfall=rainfall_data,
        climate_indices=climate_data,
    )
    print(f"Synthetic data saved to {temp_path}\n")

    # Test dataset
    print("Testing SyntheticOceanAtmosDataset...")
    ds = dataset.SyntheticOceanAtmosDataset(
        temp_path,
        split='train',
        temporal_steps=4,
        normalize=True,
    )

    print(f"Dataset size: {len(ds)}")
    assert len(ds) > 0, "Dataset should not be empty"
    print("✓ Dataset creation OK")

    # Test sample
    print("\nTesting sample loading...")
    sample = ds[0]

    print(f"Ocean shape: {sample['ocean'].shape}")
    assert sample['ocean'].shape == (3, H, W), f"Wrong ocean shape"

    print(f"Atmosphere shape: {sample['atmosphere'].shape}")
    assert sample['atmosphere'].shape == (4, 3, H, W), f"Wrong atmosphere shape"

    print(f"Climate indices shape: {sample['climate_indices'].shape}")
    assert sample['climate_indices'].shape == (3,), f"Wrong climate indices shape"

    print(f"Rainfall shape: {sample['rainfall'].shape}")
    assert sample['rainfall'].shape == (H, W), f"Wrong rainfall shape"

    print("✓ Sample loading OK")

    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        temp_path,
        batch_size=8,
        temporal_steps=4,
        num_workers=0,
    )

    print(f"Train loader size: {len(train_loader)}")
    print(f"Val loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}")

    batch = next(iter(train_loader))
    print(f"\nTrain batch ocean: {batch['ocean'].shape}")
    print(f"Train batch atmosphere: {batch['atmosphere'].shape}")
    print(f"Train batch climate: {batch['climate_indices'].shape}")
    print(f"Train batch rainfall: {batch['rainfall'].shape}")

    assert batch['ocean'].shape == (8, 3, H, W)
    assert batch['atmosphere'].shape == (8, 4, 3, H, W)
    assert batch['climate_indices'].shape == (8, 3)
    assert batch['rainfall'].shape == (8, H, W)

    print("✓ Dataloaders OK\n")

    print("=" * 60)
    print("ALL DATASET TESTS PASSED")
    print("=" * 60 + "\n")


def test_end_to_end():
    """Test end-to-end pipeline."""
    print("=" * 60)
    print("TESTING END-TO-END PIPELINE")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    print("Creating model...")
    model = models.HydroAtmosFusion(
        ocean_channels=3,
        atmo_channels=3,
        temporal_steps=4,
        num_climate_indices=3,
        feature_dim=256,
        num_attention_heads=8,
        attention_d_per_head=32,
        use_climate_indices=True,
    ).to(device)

    # Create loss and optimizer
    loss_fn = losses.CompositeLoss(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create dummy batch
    B, T, H, W = 2, 4, 64, 64
    ocean_input = torch.randn(B, 3, H, W).to(device)
    atmo_input = torch.randn(B, T, 3, H, W).to(device)
    climate_input = torch.randn(B, 3).to(device)
    target = torch.rand(B, 1, H, W).to(device) * 100

    # Test training step
    print("Testing training step...")
    model.train()

    # Forward pass
    rainfall_pred = model(ocean_input, atmo_input, climate_input)
    loss = loss_fn(rainfall_pred, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print("✓ Training step OK")

    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        rainfall_pred = model(ocean_input, atmo_input, climate_input)

    # Compute metrics
    all_metrics = metrics.evaluate_all(rainfall_pred, target, percentile=99.0)

    print("Metrics computed:")
    for name, value in all_metrics.items():
        if name != 'threshold':
            print(f"  - {name}: {value:.4f}")

    print("✓ Inference OK\n")

    print("=" * 60)
    print("END-TO-END TEST PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        test_models()
        test_losses()
        test_metrics()
        test_dataset()
        test_end_to_end()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nHydroAtmosFusion framework is ready for use.")

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
