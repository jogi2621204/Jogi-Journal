# HydroAtmosFusion Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_framework.py
```

## Quick Usage

### 1. Loading Data

```python
from hydroatmosfusion.dataset import create_dataloaders

# Create dataloaders from your .npz file
train_loader, val_loader, test_loader = create_dataloaders(
    'your_data.npz',
    batch_size=32,
    temporal_steps=4,
)
```

### 2. Creating and Training Model

```python
import torch
from hydroatmosfusion.models import HydroAtmosFusion
from hydroatmosfusion.losses import CompositeLoss

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = HydroAtmosFusion(
    ocean_channels=3,
    atmo_channels=3,
    temporal_steps=4,
    num_climate_indices=3,
).to(device)

# Loss and optimizer
loss_fn = CompositeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training step
for batch in train_loader:
    # Move to device
    ocean = batch['ocean'].to(device)
    atmosphere = batch['atmosphere'].to(device)
    climate = batch['climate_indices'].to(device)
    rainfall = batch['rainfall'].unsqueeze(1).to(device)

    # Forward pass
    pred = model(ocean, atmosphere, climate)
    loss = loss_fn(pred, rainfall)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3. Evaluation

```python
from hydroatmosfusion.metrics import evaluate_all

model.eval()
with torch.no_grad():
    pred = model(ocean, atmosphere, climate)
    metrics = evaluate_all(pred, rainfall)

print(f"CSI: {metrics['csi']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"FSS: {metrics['fss']:.4f}")
```

## Data Format

Your `.npz` file should contain:

```python
np.savez('data.npz',
    ocean=ocean_array,          # (T, H, W, 3) - SST, SSH, EKE
    atmosphere=atmo_array,       # (T, H, W, 3) - CAPE, TCWV, wind
    rainfall=rainfall_array,     # (T, H, W) - target
    climate_indices=climate_array # (T, 3) - optional climate data
)
```

Where:
- **T**: Number of timesteps
- **H, W**: Spatial dimensions
- **Ocean**: Sea Surface Temperature, Sea Surface Height, Eddy Kinetic Energy
- **Atmosphere**: Convective Available Potential Energy, Total Column Water Vapor, Wind Speed
- **Climate indices**: Any auxiliary indices (e.g., Nino34, DMI, MJO)

## Common Tasks

### Save and Load Model

```python
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model.load_state_dict(torch.load('model.pt'))
```

### Batch Prediction

```python
model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        ocean = batch['ocean'].to(device)
        atmosphere = batch['atmosphere'].to(device)
        climate = batch['climate_indices'].to(device)

        pred = model(ocean, atmosphere, climate)
        predictions.append(pred.cpu().numpy())

# Concatenate all predictions
all_predictions = np.concatenate(predictions, axis=0)
```

### Denormalize Predictions

```python
from hydroatmosfusion.dataset import SyntheticOceanAtmosDataset

# Get normalization stats
dataset = SyntheticOceanAtmosDataset('data.npz', split='train')
stats = dataset.get_normalization_stats()

# Denormalize
denorm_pred = SyntheticOceanAtmosDataset.denormalize_rainfall(pred, stats)
```

## Full Training Example

See `train_example.py` for a complete training script:

```bash
python train_example.py --data_path data.npz --epochs 20 --batch_size 32
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `models.py` | Neural network architectures |
| `losses.py` | Training loss functions |
| `metrics.py` | Evaluation metrics |
| `dataset.py` | Data loading and preprocessing |

## Tips

1. **Memory**: Reduce batch size if running out of GPU memory
2. **Speed**: Use `num_workers > 0` in DataLoader for faster data loading
3. **Reproducibility**: Set `seed` in args to get consistent results
4. **Validation**: Use evaluate_all() to track multiple metrics
5. **Checkpointing**: Save model periodically during training

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
train_loader, _, _ = create_dataloaders('data.npz', batch_size=16)

# Or reduce feature dimension
model = HydroAtmosFusion(feature_dim=128)
```

### Poor Metrics
- Check data normalization (should be z-scored)
- Increase temporal_steps if not capturing dependencies
- Adjust loss weights for emphasis on extremes

### Slow Training
```python
# Use more workers for data loading
train_loader, _, _ = create_dataloaders(
    'data.npz',
    num_workers=4,
    pin_memory=True,
)

# Use gradient checkpointing (if available)
# (See PyTorch documentation)
```

## References

- Paper equations: See README.md for mathematical references
- Loss functions: Eq. 22-25 in paper
- Metrics: Standard rainfall evaluation metrics
- Architecture: ResNet18 + Swin-like + Cross-attention

## Contact & Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Run test_framework.py to verify installation
3. Check console output and error messages
4. Review the complete training example in train_example.py
