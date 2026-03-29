"""Dataset and data loading utilities for HydroAtmosFusion framework.

Implements PyTorch Dataset classes for loading synthetic and real ocean-atmosphere
data with proper train/val/test splitting and normalization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings


class SyntheticOceanAtmosDataset(Dataset):
    """PyTorch Dataset for synthetic ocean-atmosphere rainfall data.

    Loads synthetic ocean and atmosphere data from .npz files and provides
    proper train/val/test splitting with z-score normalization per variable.

    Data structure expected in .npz file:
    - 'ocean': (T, H, W, 3) with SST, SSH, EKE
    - 'atmosphere': (T, H, W, 3) with CAPE, TCWV, wind_speed
    - 'climate_indices': (T, 3) with Nino34, DMI, MJO
    - 'rainfall': (T, H, W) with rainfall target

    where T is number of timesteps, H, W are spatial dimensions.

    Attributes:
        data_dict (Dict): Loaded data with normalized features
        split (str): Dataset split ('train', 'val', or 'test')
        temporal_steps (int): Number of temporal steps per sample
        normalizers (Dict): Normalization parameters (mean, std) per variable
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        temporal_steps: int = 4,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        test_frac: float = 0.2,
        normalize: bool = True,
        seed: int = 42,
    ):
        """Initialize SyntheticOceanAtmosDataset.

        Args:
            data_path (str): Path to .npz file containing data
            split (str): Dataset split ('train', 'val', or 'test'). Default: 'train'
            temporal_steps (int): Number of consecutive timesteps per sample. Default: 4
            train_frac (float): Fraction of data for training. Default: 0.7
            val_frac (float): Fraction of data for validation. Default: 0.1
            test_frac (float): Fraction of data for testing. Default: 0.2
            normalize (bool): Whether to apply z-score normalization. Default: True
            seed (int): Random seed for reproducible splitting. Default: 42
        """
        super().__init__()

        # Validate split argument
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        # Validate fractions sum to 1
        total_frac = train_frac + val_frac + test_frac
        if not np.isclose(total_frac, 1.0, atol=1e-5):
            raise ValueError(
                f"train_frac + val_frac + test_frac must sum to 1.0, "
                f"got {total_frac}. Fractions: {train_frac}, {val_frac}, {test_frac}"
            )

        self.data_path = data_path
        self.split = split
        self.temporal_steps = temporal_steps
        self.normalize = normalize
        self.seed = seed

        # Load raw data
        self._load_data()

        # Split data temporally
        self._split_data(train_frac, val_frac, test_frac)

        # Normalize if requested
        if normalize:
            self._normalize_data()

    def _load_data(self):
        """Load data from .npz file."""
        try:
            data = np.load(self.data_path, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.data_path}: {e}")

        # Verify required keys
        required_keys = ['ocean', 'atmosphere', 'rainfall']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in data file")

        # Load ocean data: (T, H, W, 3)
        ocean_raw = np.array(data['ocean'])
        if ocean_raw.ndim == 3:
            # Single channel ocean data, expand to 3 channels
            ocean_raw = np.stack([ocean_raw] * 3, axis=-1)
            warnings.warn("Ocean data is 1D, expanding to 3 channels")

        # Load atmosphere data: (T, H, W, 3)
        atmo_raw = np.array(data['atmosphere'])
        if atmo_raw.ndim == 3:
            # Single channel atmosphere data, expand to 3 channels
            atmo_raw = np.stack([atmo_raw] * 3, axis=-1)
            warnings.warn("Atmosphere data is 1D, expanding to 3 channels")

        # Load rainfall target: (T, H, W)
        rainfall_raw = np.array(data['rainfall'])

        # Load climate indices if available: (T, num_indices)
        if 'climate_indices' in data:
            climate_raw = np.array(data['climate_indices'])
            if climate_raw.ndim == 1:
                climate_raw = climate_raw[:, np.newaxis]
        else:
            # Create dummy climate indices if not provided
            T = ocean_raw.shape[0]
            climate_raw = np.zeros((T, 3))
            warnings.warn("climate_indices not found, using zeros")

        # Ensure consistency
        T_ocean = ocean_raw.shape[0]
        T_atmo = atmo_raw.shape[0]
        T_rainfall = rainfall_raw.shape[0]
        T_climate = climate_raw.shape[0]

        if not (T_ocean == T_atmo == T_rainfall == T_climate):
            raise ValueError(
                f"Temporal dimension mismatch: "
                f"ocean={T_ocean}, atmosphere={T_atmo}, "
                f"rainfall={T_rainfall}, climate={T_climate}"
            )

        # Store raw data
        self.ocean_raw = ocean_raw.astype(np.float32)      # (T, H, W, 3)
        self.atmo_raw = atmo_raw.astype(np.float32)        # (T, H, W, 3)
        self.rainfall_raw = rainfall_raw.astype(np.float32)  # (T, H, W)
        self.climate_raw = climate_raw.astype(np.float32)  # (T, num_climate)

        self.T, self.H, self.W = ocean_raw.shape[:3]
        self.num_climate_indices = climate_raw.shape[1]

    def _split_data(
        self,
        train_frac: float,
        val_frac: float,
        test_frac: float,
    ):
        """Split data temporally into train/val/test.

        Splits sequentially along time dimension to preserve temporal dependencies.

        Args:
            train_frac (float): Fraction for training
            val_frac (float): Fraction for validation
            test_frac (float): Fraction for testing
        """
        T = self.T
        train_end = int(T * train_frac)
        val_end = train_end + int(T * val_frac)

        indices = {
            'train': np.arange(0, train_end - self.temporal_steps + 1),
            'val': np.arange(train_end, val_end - self.temporal_steps + 1),
            'test': np.arange(val_end, T - self.temporal_steps + 1),
        }

        self.indices = indices[self.split]

        if len(self.indices) == 0:
            raise ValueError(
                f"No samples in {self.split} split. "
                f"Try reducing temporal_steps={self.temporal_steps}"
            )

    def _normalize_data(self):
        """Apply z-score normalization to all variables.

        Computes normalization statistics on training split and applies
        consistently across all splits.
        """
        # Compute statistics on training data
        self._compute_normalization_stats()

        # Apply normalization
        self.ocean_norm = self._normalize_array(self.ocean_raw, 'ocean')
        self.atmo_norm = self._normalize_array(self.atmo_raw, 'atmosphere')
        self.rainfall_norm = self._normalize_array(
            self.rainfall_raw[..., np.newaxis], 'rainfall'
        ).squeeze(-1)
        self.climate_norm = self._normalize_array(self.climate_raw, 'climate')

    def _compute_normalization_stats(self):
        """Compute normalization statistics on training split."""
        train_indices = self.indices if self.split == 'train' else None

        # For computing stats, use the full training set
        if train_indices is None:
            # Load indices from original training data
            T = self.T
            train_end = int(T * 0.7)  # Assuming default train_frac=0.7
            train_indices = np.arange(0, train_end)

        # Compute mean and std for each variable
        self.normalization_stats = {}

        # Ocean: per-channel across time, height, width
        ocean_subset = self.ocean_raw[train_indices]  # (N, H, W, 3)
        self.normalization_stats['ocean_mean'] = ocean_subset.mean(axis=(0, 1, 2), keepdims=True)
        self.normalization_stats['ocean_std'] = ocean_subset.std(axis=(0, 1, 2), keepdims=True) + 1e-7

        # Atmosphere: per-channel
        atmo_subset = self.atmo_raw[train_indices]  # (N, H, W, 3)
        self.normalization_stats['atmo_mean'] = atmo_subset.mean(axis=(0, 1, 2), keepdims=True)
        self.normalization_stats['atmo_std'] = atmo_subset.std(axis=(0, 1, 2), keepdims=True) + 1e-7

        # Rainfall: global across time, height, width
        rainfall_subset = self.rainfall_raw[train_indices]  # (N, H, W)
        self.normalization_stats['rainfall_mean'] = rainfall_subset.mean()
        self.normalization_stats['rainfall_std'] = rainfall_subset.std() + 1e-7

        # Climate indices: per-index across time
        climate_subset = self.climate_raw[train_indices]  # (N, num_climate)
        self.normalization_stats['climate_mean'] = climate_subset.mean(axis=0, keepdims=True)
        self.normalization_stats['climate_std'] = climate_subset.std(axis=0, keepdims=True) + 1e-7

    def _normalize_array(
        self,
        arr: np.ndarray,
        var_type: str,
    ) -> np.ndarray:
        """Apply z-score normalization to array using pre-computed statistics.

        Args:
            arr (np.ndarray): Array to normalize
            var_type (str): Variable type ('ocean', 'atmosphere', 'rainfall', 'climate')

        Returns:
            Normalized array
        """
        if var_type == 'ocean':
            mean = self.normalization_stats['ocean_mean']
            std = self.normalization_stats['ocean_std']
        elif var_type == 'atmosphere':
            mean = self.normalization_stats['atmo_mean']
            std = self.normalization_stats['atmo_std']
        elif var_type == 'rainfall':
            mean = self.normalization_stats['rainfall_mean']
            std = self.normalization_stats['rainfall_std']
        elif var_type == 'climate':
            mean = self.normalization_stats['climate_mean']
            std = self.normalization_stats['climate_std']
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

        return (arr - mean) / std

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Returns a dictionary with:
            - 'ocean': (3, H, W) ocean features [SST, SSH, EKE]
            - 'atmosphere': (T, 3, H, W) temporal atmosphere features [CAPE, TCWV, wind]
            - 'climate_indices': (num_climate,) climate indices [Nino34, DMI, MJO]
            - 'rainfall': (H, W) target rainfall field

        Args:
            idx (int): Sample index

        Returns:
            Dictionary with tensors
        """
        # Get timestep index
        t_start = self.indices[idx]

        # Load data (use normalized version if available)
        if self.normalize:
            ocean = self.ocean_norm[t_start]  # (H, W, 3)
            atmo_seq = self.atmo_norm[t_start:t_start + self.temporal_steps]  # (T, H, W, 3)
            rainfall = self.rainfall_norm[t_start + self.temporal_steps - 1]  # (H, W)
            climate = self.climate_norm[t_start]  # (num_climate,)
        else:
            ocean = self.ocean_raw[t_start]
            atmo_seq = self.atmo_raw[t_start:t_start + self.temporal_steps]
            rainfall = self.rainfall_raw[t_start + self.temporal_steps - 1]
            climate = self.climate_raw[t_start]

        # Convert to tensors and reorder dimensions
        # Ocean: (H, W, 3) -> (3, H, W)
        ocean = torch.from_numpy(ocean).permute(2, 0, 1)

        # Atmosphere: (T, H, W, 3) -> (T, 3, H, W)
        atmo_seq = torch.from_numpy(atmo_seq).permute(0, 3, 1, 2)

        # Rainfall: (H, W) -> (H, W)
        rainfall = torch.from_numpy(rainfall)

        # Climate: (num_climate,) -> (num_climate,)
        climate = torch.from_numpy(climate)

        sample = {
            'ocean': ocean,                  # (3, H, W)
            'atmosphere': atmo_seq,          # (T, 3, H, W)
            'climate_indices': climate,      # (num_climate,)
            'rainfall': rainfall,            # (H, W)
        }

        return sample

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics for denormalization.

        Returns:
            Dictionary with mean and std for each variable
        """
        if not self.normalize:
            warnings.warn("Normalization not applied, stats may be unreliable")

        return self.normalization_stats.copy()

    @staticmethod
    def denormalize_rainfall(
        normalized: torch.Tensor,
        stats: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        """Denormalize rainfall predictions back to original scale.

        Args:
            normalized (torch.Tensor): Normalized rainfall
            stats (Dict): Normalization statistics from get_normalization_stats()

        Returns:
            Denormalized rainfall tensor
        """
        mean = stats['rainfall_mean']
        std = stats['rainfall_std']
        return normalized * std + mean


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    temporal_steps: int = 4,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        data_path (str): Path to .npz data file
        batch_size (int): Batch size. Default: 32
        temporal_steps (int): Number of temporal steps. Default: 4
        num_workers (int): Number of data loading workers. Default: 0
        pin_memory (bool): Whether to pin memory for GPU. Default: True
        shuffle_train (bool): Whether to shuffle training data. Default: True

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SyntheticOceanAtmosDataset(
        data_path=data_path,
        split='train',
        temporal_steps=temporal_steps,
        normalize=True,
    )

    val_dataset = SyntheticOceanAtmosDataset(
        data_path=data_path,
        split='val',
        temporal_steps=temporal_steps,
        normalize=True,
    )

    test_dataset = SyntheticOceanAtmosDataset(
        data_path=data_path,
        split='test',
        temporal_steps=temporal_steps,
        normalize=True,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage - create synthetic data for testing
    print("Creating synthetic dataset for testing...")

    # Create synthetic data
    T, H, W = 100, 64, 64  # 100 timesteps, 64x64 spatial grid

    # Synthetic ocean data (SST, SSH, EKE)
    ocean_data = np.random.randn(T, H, W, 3) * 10 + np.array([25, 0.5, 100])

    # Synthetic atmosphere data (CAPE, TCWV, wind)
    atmo_data = np.random.randn(T, H, W, 3) * np.array([200, 10, 5]) + \
                np.array([2000, 50, 5])

    # Synthetic rainfall
    rainfall_data = np.abs(np.random.randn(T, H, W)) * 30 + 10

    # Synthetic climate indices
    climate_data = np.random.randn(T, 3)

    # Save to temp file
    temp_path = '/tmp/test_hydroatmos.npz'
    np.savez(
        temp_path,
        ocean=ocean_data,
        atmosphere=atmo_data,
        rainfall=rainfall_data,
        climate_indices=climate_data,
    )

    print(f"Saved synthetic data to {temp_path}")

    # Test dataset loading
    print("\nTesting dataset...")
    dataset = SyntheticOceanAtmosDataset(
        temp_path,
        split='train',
        temporal_steps=4,
        normalize=True,
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Ocean shape: {sample['ocean'].shape}")
    print(f"Atmosphere shape: {sample['atmosphere'].shape}")
    print(f"Climate indices shape: {sample['climate_indices'].shape}")
    print(f"Rainfall shape: {sample['rainfall'].shape}")

    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        temp_path,
        batch_size=8,
        temporal_steps=4,
    )

    batch = next(iter(train_loader))
    print(f"Train batch ocean shape: {batch['ocean'].shape}")
    print(f"Train batch atmosphere shape: {batch['atmosphere'].shape}")
    print(f"Train batch climate shape: {batch['climate_indices'].shape}")
    print(f"Train batch rainfall shape: {batch['rainfall'].shape}")
