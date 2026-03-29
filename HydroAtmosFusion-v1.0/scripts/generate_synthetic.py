"""
Generate physics-consistent synthetic dataset for HydroAtmosFusion.

This module generates a synthetic dataset following the equations and parameters
described in the HydroAtmosFusion paper. The dataset is deterministic with a
fixed random seed and includes ocean, atmospheric, and climate variables.

Equations:
- SST(x,y,t): Sea surface temperature with spatial gaussian and seasonal cycle
- SSH: Sea surface height from steric effect and eddies
- CAPE: Convective available potential energy
- TCWV: Total column water vapor
- P: Rainfall from CAPE, TCWV, and cloud cover
- u,v: Wind fields from geostrophic balance
- EKE: Eddy kinetic energy
- Climate indices: ONI, DMI, RMM1, RMM2
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate physics-consistent synthetic oceanographic and atmospheric data."""

    def __init__(self, seed: int = 42, grid_size: int = 64, num_days: int = 365):
        """
        Initialize the synthetic data generator.

        Args:
            seed: Random seed for reproducibility
            grid_size: Size of the spatial grid (grid_size x grid_size)
            num_days: Number of days to generate
        """
        self.seed = seed
        np.random.seed(seed)
        self.grid_size = grid_size
        self.num_days = num_days

        # Domain specifications
        self.lon_min, self.lon_max = 100.0, 130.0
        self.lat_min, self.lat_max = -10.0, 10.0

        # Create coordinate grids
        self.lon = np.linspace(self.lon_min, self.lon_max, grid_size)
        self.lat = np.linspace(self.lat_min, self.lat_max, grid_size)
        self.X, self.Y = np.meshgrid(self.lon, self.lat)
        self.time = np.arange(num_days)

        # SST parameters (Eq. 12)
        self.SST0 = 28.5      # Base SST in °C
        self.A1 = 2.0         # Gaussian amplitude in °C
        self.sigma = 15       # Gaussian width in grid points
        self.A2 = 1.5         # Seasonal amplitude in °C
        self.sst_noise_std = 0.3

        # Gaussian center
        self.x0_idx = grid_size // 2
        self.y0_idx = grid_size // 2
        self.x0 = self.lon[self.x0_idx]
        self.y0 = self.lat[self.y0_idx]

        # SSH parameters (Eq. 13)
        self.alpha_s = 2.1e-4  # Steric coefficient K^-1
        self.h_ml = 50.0       # Mixed layer depth in meters
        self.rho_0 = 1025.0    # Reference seawater density kg/m^3

        # CAPE parameters (Eq. 14)
        self.CAPE0 = 1200.0    # Reference CAPE in J/kg
        self.alpha_cape = 0.06
        self.Tref = 28.0       # Reference temperature in °C

        # TCWV parameters (Eq. 15)
        self.TCWV0 = 55.0      # Reference TCWV in kg/m^2
        self.gamma = 4.5       # TCWV sensitivity to SST in kg/m^2/K

        # Rainfall parameters (Eq. 16)
        self.beta1 = 3.5
        self.beta2 = 1.5
        self.beta3 = 2.0
        self.beta4 = 0.4

        # Extreme event parameters
        self.extreme_prob = 0.02
        self.extreme_percentile = 99

        # Climate index parameters
        self.oni_period = 365 * 3.5  # ~3.5 year cycle

        logger.info(f"Initialized SyntheticDataGenerator: grid_size={grid_size}, "
                   f"num_days={num_days}, seed={seed}")

    def _generate_sst(self) -> np.ndarray:
        """
        Generate sea surface temperature field (Eq. 12).

        SST(x,y,t) = SST0 + A1*exp(-((x-x0)²+(y-y0)²)/(2σ²))
                     + A2*sin(2πt/365) + ε

        Returns:
            SST array of shape (num_days, grid_size, grid_size)
        """
        logger.info("Generating SST field...")
        sst = np.zeros((self.num_days, self.grid_size, self.grid_size))

        # Spatial component: Gaussian anomaly
        x_dist = (self.X - self.x0) ** 2
        y_dist = (self.Y - self.y0) ** 2
        gaussian = self.A1 * np.exp(-(x_dist + y_dist) / (2 * self.sigma ** 2))

        # Temporal component
        for t in range(self.num_days):
            seasonal = self.A2 * np.sin(2 * np.pi * t / 365.0)
            noise = np.random.normal(0, self.sst_noise_std,
                                    (self.grid_size, self.grid_size))
            sst[t, :, :] = self.SST0 + gaussian + seasonal + noise

        logger.info(f"SST range: {sst.min():.2f} to {sst.max():.2f} °C")
        return sst

    def _generate_ssh(self, sst: np.ndarray) -> np.ndarray:
        """
        Generate sea surface height from steric height and eddies (Eq. 13).

        SSH = αs*(SST-SST0)*h_ml + SSH_edd

        Args:
            sst: SST array of shape (num_days, grid_size, grid_size)

        Returns:
            SSH array of shape (num_days, grid_size, grid_size)
        """
        logger.info("Generating SSH field...")
        ssh = np.zeros((self.num_days, self.grid_size, self.grid_size))

        # Steric component
        sst_anom = sst - self.SST0
        ssh_steric = self.alpha_s * sst_anom * self.h_ml

        # Eddy component: red noise
        for t in range(self.num_days):
            if t == 0:
                ssh_edd = np.random.normal(0, 0.05, (self.grid_size, self.grid_size))
            else:
                # AR(1) process for temporal correlation
                ssh_edd = 0.7 * ssh[t-1, :, :] + 0.3 * np.random.normal(
                    0, 0.05, (self.grid_size, self.grid_size))
            ssh[t, :, :] = ssh_steric[t, :, :] + ssh_edd

        logger.info(f"SSH range: {ssh.min():.4f} to {ssh.max():.4f} m")
        return ssh

    def _generate_cape(self, sst: np.ndarray) -> np.ndarray:
        """
        Generate CAPE field (Eq. 14).

        CAPE = CAPE0 * exp(α*(SST-Tref)/Tref)

        Args:
            sst: SST array of shape (num_days, grid_size, grid_size)

        Returns:
            CAPE array of shape (num_days, grid_size, grid_size)
        """
        logger.info("Generating CAPE field...")
        sst_normalized = (sst - self.Tref) / self.Tref
        cape = self.CAPE0 * np.exp(self.alpha_cape * sst_normalized)
        logger.info(f"CAPE range: {cape.min():.1f} to {cape.max():.1f} J/kg")
        return cape

    def _generate_tcwv(self, sst: np.ndarray) -> np.ndarray:
        """
        Generate total column water vapor (Eq. 15).

        TCWV = TCWV0 + γ*(SST-Tref)

        Args:
            sst: SST array of shape (num_days, grid_size, grid_size)

        Returns:
            TCWV array of shape (num_days, grid_size, grid_size)
        """
        logger.info("Generating TCWV field...")
        tcwv = self.TCWV0 + self.gamma * (sst - self.Tref)
        # Ensure physical bounds
        tcwv = np.clip(tcwv, 10, 80)
        logger.info(f"TCWV range: {tcwv.min():.1f} to {tcwv.max():.1f} kg/m²")
        return tcwv

    def _generate_cloud_cover(self) -> np.ndarray:
        """
        Generate cloud cover field (used in rainfall equation).

        Returns:
            Cloud cover array of shape (num_days, grid_size, grid_size) in [0, 1]
        """
        logger.info("Generating cloud cover field...")
        # Cloud cover with temporal and spatial coherence
        cloud = np.zeros((self.num_days, self.grid_size, self.grid_size))

        for t in range(self.num_days):
            if t == 0:
                cloud[t, :, :] = np.random.uniform(0.3, 0.8,
                                                   (self.grid_size, self.grid_size))
            else:
                # AR(1) process
                cloud[t, :, :] = (0.7 * cloud[t-1, :, :] +
                                 0.3 * np.random.uniform(0.3, 0.8,
                                                        (self.grid_size, self.grid_size)))

        logger.info(f"Cloud cover range: {cloud.min():.2f} to {cloud.max():.2f}")
        return cloud

    def _generate_rainfall(self, cape: np.ndarray, tcwv: np.ndarray,
                          cloud: np.ndarray) -> np.ndarray:
        """
        Generate rainfall field (Eq. 16).

        P = β1*(CAPE/CAPE0)^β2 * (TCWV/TCWV0)^β3 * (1+β4*C)

        Args:
            cape: CAPE array
            tcwv: TCWV array
            cloud: Cloud cover array

        Returns:
            Rainfall array of shape (num_days, grid_size, grid_size)
        """
        logger.info("Generating rainfall field...")

        cape_norm = cape / self.CAPE0
        tcwv_norm = tcwv / self.TCWV0

        rainfall = (self.beta1 *
                   np.power(cape_norm, self.beta2) *
                   np.power(tcwv_norm, self.beta3) *
                   (1.0 + self.beta4 * cloud))

        # Inject extreme events
        P99 = np.percentile(rainfall, self.extreme_percentile)
        extreme_mask = np.random.random(rainfall.shape) < self.extreme_prob
        rainfall[extreme_mask] = np.random.uniform(P99, P99 * 1.5,
                                                   np.sum(extreme_mask))

        rainfall = np.clip(rainfall, 0, None)  # Ensure non-negative
        logger.info(f"Rainfall range: {rainfall.min():.2f} to {rainfall.max():.2f} mm/day")
        logger.info(f"P99: {P99:.2f} mm/day, Extreme events: {np.sum(extreme_mask)} points")

        return rainfall

    def _generate_wind_fields(self, ssh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate geostrophic wind fields from SSH.

        u = -(g/f) * ∂SSH/∂y
        v = (g/f) * ∂SSH/∂x

        Args:
            ssh: SSH array

        Returns:
            Tuple of (u, v) wind component arrays
        """
        logger.info("Generating geostrophic wind fields...")

        # Coriolis parameter (approximate for tropics)
        coriolis = 2 * 7.27e-5 * np.sin(np.radians(self.Y))
        coriolis = np.where(np.abs(coriolis) < 1e-5, 1e-5, coriolis)

        g = 9.81  # Gravity

        u = np.zeros_like(ssh)
        v = np.zeros_like(ssh)

        for t in range(self.num_days):
            # Compute gradients with proper spacing
            dlon = np.radians(self.lon[1] - self.lon[0])
            dlat = np.radians(self.lat[1] - self.lat[0])

            # Convert to meters (at mean latitude)
            dy = dlat * 6.371e6
            dx = dlon * 6.371e6 * np.cos(np.radians(self.Y))

            dy_ssh = np.gradient(ssh[t, :, :], axis=0) / dy
            dx_ssh = np.gradient(ssh[t, :, :], axis=1) / dx

            u[t, :, :] = -(g / coriolis) * dy_ssh
            v[t, :, :] = (g / coriolis) * dx_ssh

        # Add turbulent component
        u_turb = np.random.normal(0, 0.1, u.shape)
        v_turb = np.random.normal(0, 0.1, v.shape)
        u = u + u_turb
        v = v + v_turb

        u = np.clip(u, -10, 10)
        v = np.clip(v, -10, 10)

        logger.info(f"U-wind range: {u.min():.2f} to {u.max():.2f} m/s")
        logger.info(f"V-wind range: {v.min():.2f} to {v.max():.2f} m/s")

        return u, v

    def _generate_eke(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Generate eddy kinetic energy.

        EKE = 0.5 * (u'² + v'²)

        Args:
            u: U-wind component
            v: V-wind component

        Returns:
            EKE array
        """
        logger.info("Generating EKE field...")

        # Compute wind anomalies
        u_mean = u.mean(axis=0)
        v_mean = v.mean(axis=0)

        u_prime = u - u_mean
        v_prime = v - v_mean

        eke = 0.5 * (u_prime ** 2 + v_prime ** 2)
        logger.info(f"EKE range: {eke.min():.2f} to {eke.max():.2f} m²/s²")

        return eke

    def _generate_oni(self) -> np.ndarray:
        """
        Generate Oceanic Niño Index (ONI) climate index.

        ONI follows a sinusoidal pattern with ~3.5 year period.

        Returns:
            ONI array of shape (num_days,)
        """
        logger.info("Generating ONI index...")
        oni = 1.0 * np.sin(2 * np.pi * self.time / self.oni_period)
        oni = oni + np.random.normal(0, 0.2, self.num_days)
        logger.info(f"ONI range: {oni.min():.2f} to {oni.max():.2f}")
        return oni

    def _generate_dmi(self) -> np.ndarray:
        """
        Generate Dipole Mode Index (DMI) climate index.

        DMI is the sea surface temperature gradient between western and eastern Indian Ocean.

        Returns:
            DMI array of shape (num_days,)
        """
        logger.info("Generating DMI index...")
        # Annual and 2-4 year oscillations
        dmi = (0.8 * np.sin(2 * np.pi * self.time / 365.0) +
               0.6 * np.sin(2 * np.pi * self.time / 1200.0) +
               np.random.normal(0, 0.15, self.num_days))
        logger.info(f"DMI range: {dmi.min():.2f} to {dmi.max():.2f}")
        return dmi

    def _generate_rmm(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Real-time Multivariate MJO Index (RMM1, RMM2).

        RMM indices represent the Madden-Julian Oscillation phase/amplitude.

        Returns:
            Tuple of (RMM1, RMM2) arrays of shape (num_days,)
        """
        logger.info("Generating RMM indices...")

        # MJO period ~30-60 days, typically 40 days
        mjo_period = 40.0

        rmm1 = (np.sin(2 * np.pi * self.time / mjo_period) +
               0.3 * np.sin(2 * np.pi * self.time / 365.0) +
               np.random.normal(0, 0.1, self.num_days))

        rmm2 = (np.cos(2 * np.pi * self.time / mjo_period) +
               0.3 * np.cos(2 * np.pi * self.time / 365.0) +
               np.random.normal(0, 0.1, self.num_days))

        logger.info(f"RMM1 range: {rmm1.min():.2f} to {rmm1.max():.2f}")
        logger.info(f"RMM2 range: {rmm2.min():.2f} to {rmm2.max():.2f}")

        return rmm1, rmm2

    def generate(self) -> Dict[str, np.ndarray]:
        """
        Generate complete synthetic dataset.

        Returns:
            Dictionary containing all generated fields and metadata
        """
        logger.info("=" * 70)
        logger.info("Starting synthetic data generation")
        logger.info("=" * 70)

        # Generate ocean variables
        sst = self._generate_sst()
        ssh = self._generate_ssh(sst)

        # Generate atmospheric variables
        cape = self._generate_cape(sst)
        tcwv = self._generate_tcwv(sst)
        cloud = self._generate_cloud_cover()
        rainfall = self._generate_rainfall(cape, tcwv, cloud)

        # Generate wind fields and EKE
        u, v = self._generate_wind_fields(ssh)
        wind_speed = np.sqrt(u ** 2 + v ** 2)
        eke = self._generate_eke(u, v)

        # Generate climate indices
        oni = self._generate_oni()
        dmi = self._generate_dmi()
        rmm1, rmm2 = self._generate_rmm()

        logger.info("=" * 70)
        logger.info("Synthetic data generation complete")
        logger.info("=" * 70)

        return {
            # Ocean variables
            'sst': sst,
            'ssh': ssh,
            'eke': eke,
            # Atmospheric variables
            'cape': cape,
            'tcwv': tcwv,
            'cloud': cloud,
            'rainfall': rainfall,
            # Wind components
            'u': u,
            'v': v,
            'wind_speed': wind_speed,
            # Climate indices
            'oni': oni,
            'dmi': dmi,
            'rmm1': rmm1,
            'rmm2': rmm2,
            # Metadata
            'lon': self.lon,
            'lat': self.lat,
            'time': self.time,
            'domain': {
                'lon_min': self.lon_min,
                'lon_max': self.lon_max,
                'lat_min': self.lat_min,
                'lat_max': self.lat_max,
            },
            'metadata': {
                'seed': self.seed,
                'grid_size': self.grid_size,
                'num_days': self.num_days,
                'parameters': {
                    'SST0': self.SST0,
                    'A1': self.A1,
                    'sigma': self.sigma,
                    'A2': self.A2,
                    'alpha_s': self.alpha_s,
                    'h_ml': self.h_ml,
                    'CAPE0': self.CAPE0,
                    'TCWV0': self.TCWV0,
                }
            }
        }

    def save(self, data: Dict[str, np.ndarray], output_path: str) -> None:
        """
        Save generated data to NPZ file.

        Args:
            data: Dictionary of arrays to save
            output_path: Path to output .npz file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving synthetic data to {output_path}")

        # Convert metadata to numpy array format for saving
        np.savez(output_path, **data)

        file_size = output_path.stat().st_size / (1024 ** 2)  # MB
        logger.info(f"Successfully saved synthetic data ({file_size:.1f} MB)")


def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description='Generate physics-consistent synthetic dataset for HydroAtmosFusion'
    )
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--grid_size', type=int, default=64,
                       help='Size of the spatial grid')
    parser.add_argument('--num_days', type=int, default=365,
                       help='Number of days to generate')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--output_file', type=str,
                       default='synthetic_ocean_atmos.npz',
                       help='Output filename')

    args = parser.parse_args()

    # Create generator
    generator = SyntheticDataGenerator(
        seed=args.seed,
        grid_size=args.grid_size,
        num_days=args.num_days
    )

    # Generate data
    data = generator.generate()

    # Save data
    output_path = Path(args.output_dir) / args.output_file
    generator.save(data, str(output_path))

    logger.info(f"Generated dataset keys: {list(data.keys())}")


if __name__ == '__main__':
    main()
