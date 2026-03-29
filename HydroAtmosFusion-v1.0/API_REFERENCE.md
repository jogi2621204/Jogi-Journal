# HydroAtmosFusion API Reference

Complete API documentation for the HydroAtmosFusion framework.

## Table of Contents

1. [Models (`models.py`)](#models)
2. [Losses (`losses.py`)](#losses)
3. [Metrics (`metrics.py`)](#metrics)
4. [Dataset (`dataset.py`)](#dataset)

---

## Models (`models.py`)

### SqueezeExcitationBlock

Squeeze-and-Excitation block for channel attention.

```python
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 4)
```

**Parameters:**
- `channels` (int): Number of input channels
- `reduction_ratio` (int): Reduction ratio for FC layers. Default: 4

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass

**Shape:**
- Input: (B, C, H, W)
- Output: (B, C, H, W)

---

### OceanEncoder

ResNet-18 based ocean state encoder with SE blocks.

```python
class OceanEncoder(nn.Module):
    def __init__(self, num_channels: int = 3, output_channels: int = 256)
```

**Parameters:**
- `num_channels` (int): Number of input channels (default: 3 for SST, SSH, EKE)
- `output_channels` (int): Number of output channels (default: 256)

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass

**Shape:**
- Input: (B, 3, H, W)
- Output: (B, 256, H/8, W/8)

**Example:**
```python
encoder = OceanEncoder(num_channels=3, output_channels=256)
ocean_input = torch.randn(4, 3, 64, 64)
ocean_features = encoder(ocean_input)  # (4, 256, 8, 8)
```

---

### ConvGRUCell

Convolutional GRU cell for spatiotemporal processing.

```python
class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3)
```

**Parameters:**
- `input_channels` (int): Number of input channels
- `hidden_channels` (int): Number of hidden channels
- `kernel_size` (int): Convolution kernel size. Default: 3

**Methods:**
- `forward(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor`: Forward pass

**Shape:**
- Input x: (B, C_in, H, W)
- Input h: (B, C_hidden, H, W)
- Output: (B, C_hidden, H, W)

---

### AtmosphereEncoder

Hierarchical atmosphere encoder with ConvGRU temporal gating.

```python
class AtmosphereEncoder(nn.Module):
    def __init__(self, num_channels: int = 3, output_channels: int = 256,
                 temporal_steps: int = 4)
```

**Parameters:**
- `num_channels` (int): Input channels (default: 3 for CAPE, TCWV, wind)
- `output_channels` (int): Output channels (default: 256)
- `temporal_steps` (int): Number of temporal steps (default: 4)

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass

**Shape:**
- Input: (B, T, 3, H, W)
- Output: (B, 256, H/4, W/4)

**Example:**
```python
encoder = AtmosphereEncoder(num_channels=3, output_channels=256, temporal_steps=4)
atmo_input = torch.randn(4, 4, 3, 64, 64)  # 4 timesteps
atmo_features = encoder(atmo_input)  # (4, 256, 16, 16)
```

---

### MultiHeadCrossAttention

Multi-head cross-attention module.

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8,
                 d_per_head: int = 32)
```

**Parameters:**
- `query_dim` (int): Query feature dimension
- `key_dim` (int): Key-value feature dimension
- `num_heads` (int): Number of attention heads (default: 8)
- `d_per_head` (int): Dimension per head (default: 32)

**Methods:**
- `forward(query, key, value) -> torch.Tensor`: Attention forward pass

**Shape:**
- Query: (B, L_query, D_query)
- Key: (B, L_key, D_key)
- Value: (B, L_value, D_key)
- Output: (B, L_query, D_query)

---

### BidirectionalCrossModalAttention

Bidirectional cross-modal attention fusion (Eq. 20-21).

```python
class BidirectionalCrossModalAttention(nn.Module):
    def __init__(self, feature_dim: int = 256, num_heads: int = 8,
                 d_per_head: int = 32)
```

**Parameters:**
- `feature_dim` (int): Feature dimension (default: 256)
- `num_heads` (int): Attention heads (default: 8)
- `d_per_head` (int): Dimension per head (default: 32)

**Methods:**
- `forward(ocean_feat, atmo_feat) -> torch.Tensor`: Bidirectional fusion

**Shape:**
- Ocean features: (B, 256, H/4, W/4)
- Atmosphere features: (B, 256, H/4, W/4)
- Output: (B, 512, H/4, W/4)

**Example:**
```python
fusion = BidirectionalCrossModalAttention(feature_dim=256, num_heads=8)
ocean_feat = torch.randn(4, 256, 16, 16)
atmo_feat = torch.randn(4, 256, 16, 16)
fused = fusion(ocean_feat, atmo_feat)  # (4, 512, 16, 16)
```

---

### ClimateIndexEncoder

MLP-based climate index encoder.

```python
class ClimateIndexEncoder(nn.Module):
    def __init__(self, num_indices: int = 3, output_dim: int = 64)
```

**Parameters:**
- `num_indices` (int): Number of climate indices (default: 3)
- `output_dim` (int): Output feature dimension (default: 64)

**Methods:**
- `forward(climate_indices, spatial_shape) -> torch.Tensor`: Encode and broadcast

**Shape:**
- Climate indices: (B, num_indices)
- Spatial shape: (H, W)
- Output: (B, output_dim, H, W)

---

### PredictionHead

Output prediction head for rainfall field.

```python
class PredictionHead(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 1)
```

**Parameters:**
- `input_channels` (int): Input feature channels
- `output_channels` (int): Output channels (default: 1)

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass

**Shape:**
- Input: (B, C_in, H, W)
- Output: (B, 1, H, W) with sigmoid activation

---

### HydroAtmosFusion

Main integrated model combining all components.

```python
class HydroAtmosFusion(nn.Module):
    def __init__(self, ocean_channels: int = 3, atmo_channels: int = 3,
                 temporal_steps: int = 4, num_climate_indices: int = 3,
                 feature_dim: int = 256, num_attention_heads: int = 8,
                 attention_d_per_head: int = 32, use_climate_indices: bool = True)
```

**Parameters:**
- `ocean_channels` (int): Ocean input channels (default: 3)
- `atmo_channels` (int): Atmosphere input channels (default: 3)
- `temporal_steps` (int): Temporal steps (default: 4)
- `num_climate_indices` (int): Climate indices count (default: 3)
- `feature_dim` (int): Feature dimension (default: 256)
- `num_attention_heads` (int): Attention heads (default: 8)
- `attention_d_per_head` (int): Dimension per head (default: 32)
- `use_climate_indices` (bool): Include climate indices (default: True)

**Methods:**
- `forward(ocean, atmosphere, climate_indices=None) -> torch.Tensor`: Forward pass

**Shape:**
- Ocean: (B, 3, H, W)
- Atmosphere: (B, T, 3, H, W)
- Climate indices: (B, num_climate) (optional)
- Output: (B, 1, H, W) in [0, 1]

**Example:**
```python
model = HydroAtmosFusion(use_climate_indices=True)
ocean = torch.randn(4, 3, 64, 64)
atmo = torch.randn(4, 4, 3, 64, 64)
climate = torch.randn(4, 3)

rainfall = model(ocean, atmo, climate)  # (4, 1, 16, 16)
```

---

## Losses (`losses.py`)

### WeightedMSELoss

Weighted MSE with emphasis on extreme rainfall (Eq. 22).

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, lambda_weight: float = 5.0, percentile: float = 99.0)
```

**Parameters:**
- `lambda_weight` (float): Weight multiplier for extremes (default: 5)
- `percentile` (float): Percentile threshold (default: 99)

**Methods:**
- `forward(pred, true) -> torch.Tensor`: Loss computation

**Formula:** w(y) = 1 + lambda * I(y > P99)

---

### FocalCSILoss

Focal loss applied to Critical Success Index (Eq. 23).

```python
class FocalCSILoss(nn.Module):
    def __init__(self, threshold: Optional[float] = None, gamma: float = 2.0,
                 eps: float = 1e-7)
```

**Parameters:**
- `threshold` (float): Event detection threshold (default: None, uses P99)
- `gamma` (float): Focal loss parameter (default: 2)
- `eps` (float): Numerical stability (default: 1e-7)

**Methods:**
- `forward(pred, true) -> torch.Tensor`: Loss computation

**Formula:** L = -(1-CSI)^gamma * log(CSI + eps)

---

### FSSLoss

Fractions Skill Score loss (Eq. 24).

```python
class FSSLoss(nn.Module):
    def __init__(self, radius: int = 5, threshold: Optional[float] = None)
```

**Parameters:**
- `radius` (int): Neighborhood radius (default: 5)
- `threshold` (float): Event threshold (default: None, uses P99)

**Methods:**
- `forward(pred, true) -> torch.Tensor`: Loss computation (returns 1 - FSS)

**Formula:** FSS = 1 - MSE_neighborhood / MSE_perfect

---

### CompositeLoss

Composite loss combining all components (Eq. 25).

```python
class CompositeLoss(nn.Module):
    def __init__(self, alpha: float = 0.4, beta: float = 0.35, gamma: float = 0.25,
                 wmse_lambda: float = 5.0, wmse_percentile: float = 99.0,
                 fss_radius: int = 5, threshold: Optional[float] = None)
```

**Parameters:**
- `alpha` (float): WMSE weight (default: 0.4)
- `beta` (float): Focal CSI weight (default: 0.35)
- `gamma` (float): FSS weight (default: 0.25)
- `wmse_lambda` (float): WMSE lambda (default: 5.0)
- `wmse_percentile` (float): WMSE percentile (default: 99.0)
- `fss_radius` (int): FSS radius (default: 5)
- `threshold` (float): Event threshold (default: None)

**Methods:**
- `forward(pred, true, return_components=False) -> Union[torch.Tensor, Tuple]`

**Formula:** L = alpha*L_WMSE + beta*L_FocalCSI + gamma*L_FSS

**Example:**
```python
loss_fn = CompositeLoss(alpha=0.4, beta=0.35, gamma=0.25)
loss, components = loss_fn(pred, target, return_components=True)

print(f"Total: {loss.item():.4f}")
print(f"WMSE: {components['wmse']:.4f}")
print(f"Focal CSI: {components['focal_csi']:.4f}")
print(f"FSS: {components['fss']:.4f}")
```

---

## Metrics (`metrics.py`)

### compute_rmse

Root Mean Squared Error.

```python
def compute_rmse(pred: torch.Tensor, true: torch.Tensor) -> float
```

**Returns:** RMSE value (float)

---

### compute_mae

Mean Absolute Error.

```python
def compute_mae(pred: torch.Tensor, true: torch.Tensor) -> float
```

**Returns:** MAE value (float)

---

### compute_correlation

Spatial correlation coefficient (Pearson).

```python
def compute_correlation(pred: torch.Tensor, true: torch.Tensor) -> float
```

**Returns:** Correlation in [-1, 1]

---

### compute_csi

Critical Success Index.

```python
def compute_csi(pred: torch.Tensor, true: torch.Tensor,
                threshold: Optional[float] = None,
                percentile: float = 99.0) -> float
```

**Parameters:**
- `pred`: Predictions (B, 1, H, W) or (B, H, W)
- `true`: Targets (B, 1, H, W) or (B, H, W)
- `threshold`: Event threshold (default: None, uses percentile)
- `percentile`: Percentile for auto threshold (default: 99)

**Returns:** CSI in [0, 1]

**Formula:** CSI = TP / (TP + FP + FN)

---

### compute_edi

Extremal Dependence Index.

```python
def compute_edi(pred: torch.Tensor, true: torch.Tensor,
                threshold: Optional[float] = None,
                percentile: float = 99.0) -> float
```

**Returns:** EDI in [0, 1]

---

### compute_fss

Fractions Skill Score.

```python
def compute_fss(pred: torch.Tensor, true: torch.Tensor,
                threshold: Optional[float] = None,
                percentile: float = 99.0,
                radius: int = 5) -> float
```

**Parameters:**
- `radius` (int): Search radius (default: 5)

**Returns:** FSS in [0, 1]

---

### evaluate_all

Compute all metrics at once.

```python
def evaluate_all(pred: torch.Tensor, true: torch.Tensor,
                 percentile: float = 99.0,
                 threshold: Optional[float] = None,
                 fss_radius: int = 5) -> Dict[str, float]
```

**Returns:** Dictionary with all metrics

**Example:**
```python
metrics = evaluate_all(pred, target, percentile=99.0)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"Correlation: {metrics['correlation']:.4f}")
print(f"CSI: {metrics['csi']:.4f}")
print(f"EDI: {metrics['edi']:.4f}")
print(f"FSS: {metrics['fss']:.4f}")
```

---

## Dataset (`dataset.py`)

### SyntheticOceanAtmosDataset

PyTorch Dataset for ocean-atmosphere rainfall data.

```python
class SyntheticOceanAtmosDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train',
                 temporal_steps: int = 4, train_frac: float = 0.7,
                 val_frac: float = 0.1, test_frac: float = 0.2,
                 normalize: bool = True, seed: int = 42)
```

**Parameters:**
- `data_path` (str): Path to .npz file
- `split` (str): 'train', 'val', or 'test'
- `temporal_steps` (int): Consecutive timesteps per sample (default: 4)
- `train_frac` (float): Training fraction (default: 0.7)
- `val_frac` (float): Validation fraction (default: 0.1)
- `test_frac` (float): Test fraction (default: 0.2)
- `normalize` (bool): Apply z-score normalization (default: True)
- `seed` (int): Random seed (default: 42)

**Methods:**
- `__len__() -> int`: Number of samples
- `__getitem__(idx) -> Dict[str, torch.Tensor]`: Get sample
- `get_normalization_stats() -> Dict`: Get normalization parameters
- `denormalize_rainfall(normalized, stats) -> torch.Tensor`: Denormalize

**Returns (from `__getitem__`):**
```python
{
    'ocean': (3, H, W),           # SST, SSH, EKE
    'atmosphere': (T, 3, H, W),   # CAPE, TCWV, wind
    'climate_indices': (num_climate,),
    'rainfall': (H, W),
}
```

**Example:**
```python
dataset = SyntheticOceanAtmosDataset(
    'data.npz',
    split='train',
    temporal_steps=4,
    normalize=True,
)

sample = dataset[0]
print(sample['ocean'].shape)        # (3, 64, 64)
print(sample['atmosphere'].shape)   # (4, 3, 64, 64)
print(sample['rainfall'].shape)     # (64, 64)
```

---

### create_dataloaders

Create train/val/test DataLoaders.

```python
def create_dataloaders(data_path: str, batch_size: int = 32,
                       temporal_steps: int = 4, num_workers: int = 0,
                       pin_memory: bool = True,
                       shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]
```

**Returns:** (train_loader, val_loader, test_loader)

**Example:**
```python
train_loader, val_loader, test_loader = create_dataloaders(
    'data.npz',
    batch_size=32,
    temporal_steps=4,
    num_workers=4,
)

for batch in train_loader:
    ocean = batch['ocean']         # (32, 3, H, W)
    atmosphere = batch['atmosphere'] # (32, 4, 3, H, W)
    rainfall = batch['rainfall']   # (32, H, W)
```

---

## Input/Output Specifications

### Data Format (.npz)

```python
np.savez('data.npz',
    # Required
    ocean=np.ndarray,          # Shape: (T, H, W, 3)
                               # Channels: [SST, SSH, EKE]

    atmosphere=np.ndarray,     # Shape: (T, H, W, 3)
                               # Channels: [CAPE, TCWV, wind_speed]

    rainfall=np.ndarray,       # Shape: (T, H, W)
                               # Target rainfall

    # Optional
    climate_indices=np.ndarray # Shape: (T, num_indices)
                               # Auxiliary climate data
)
```

### Model Input/Output

**Input:**
- Ocean: (B, 3, H, W) - normalized SST, SSH, EKE
- Atmosphere: (B, T, 3, H, W) - normalized CAPE, TCWV, wind
- Climate indices: (B, num_climate) - optional normalized indices

**Output:**
- Rainfall prediction: (B, 1, H/4, W/4) with values in [0, 1]

---

## Type Hints

All functions use proper type hints for IDE support:

```python
def function(input_tensor: torch.Tensor,
            param: int = 5) -> Dict[str, float]:
```

---

## Error Handling

Common exceptions:

- `FileNotFoundError`: Data file not found
- `ValueError`: Invalid split, mismatched dimensions, invalid weights
- `RuntimeError`: CUDA out of memory, shape mismatch

---

## Performance Tips

1. **Batch Size**: 32 for GPU (RTX 3080), 8-16 for smaller GPUs
2. **Workers**: `num_workers=4` on multi-core systems
3. **Memory**: ~2.5GB for batch size 32 at 64x64 resolution
4. **Speed**: ~100ms per batch on RTX 3080

---

## References

- Paper equations: Implemented as specified
- Standard ML practices: PyTorch conventions followed
- Data normalization: Z-score (mean=0, std=1)
