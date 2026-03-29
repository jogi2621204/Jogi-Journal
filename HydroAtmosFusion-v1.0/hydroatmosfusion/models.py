"""PyTorch models for HydroAtmosFusion framework.

Implements all neural network components for multimodal extreme rainfall prediction:
- OceanEncoder: ResNet-18 backbone with Squeeze-and-Excitation attention
- AtmosphereEncoder: Swin-Transformer-inspired hierarchical encoder with ConvGRU
- BidirectionalCrossModalAttention: Bidirectional cross-attention fusion (Eq. 20-21)
- PredictionHead: Output head for rainfall field prediction
- ClimateIndexEncoder: MLP encoder for auxiliary climate indices
- HydroAtmosFusion: Main integrated model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Optional, Tuple, Dict
import math


class SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention (Eq. 17).

    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels using global average pooling followed
    by two fully-connected layers.

    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the bottleneck. Default: 4.
    """

    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying channel attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W) with channel attention applied
        """
        # Global average pooling: (B, C, H, W) -> (B, C)
        squeeze = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        # Excitation: FC layers with ReLU and Sigmoid
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))

        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        excitation = excitation.view(x.size(0), x.size(1), 1, 1)

        # Scale channels
        return x * excitation


class OceanEncoder(nn.Module):
    """Ocean state encoder using ResNet-18 with Squeeze-and-Excitation attention.

    Encodes ocean features (SST, SSH, EKE) using a ResNet-18 backbone with
    SE blocks for channel attention (Eq. 17). Processes 3-channel ocean input
    to 256-dimensional feature map at 1/4 spatial resolution.

    Input channels: 3 (SST, SSH, EKE)
    Output channels: 256
    Spatial reduction: 1/4 (H/4, W/4)
    """

    def __init__(self, num_channels: int = 3, output_channels: int = 256):
        """Initialize OceanEncoder.

        Args:
            num_channels (int): Number of input channels. Default: 3 (SST, SSH, EKE)
            output_channels (int): Number of output channels. Default: 256
        """
        super().__init__()

        # Load pretrained ResNet-18 backbone
        resnet = resnet18(pretrained=False)

        # Adapt first layer to accept variable number of input channels
        if num_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Copy weights from original conv if num_channels == 3, else initialize
            if num_channels == 3:
                resnet.conv1.weight.data = original_conv.weight.data

        # Extract backbone layers up to layer3 (stride=16 from input)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Add SE blocks for each layer
        self.se1 = SqueezeExcitationBlock(64, reduction_ratio=4)
        self.se2 = SqueezeExcitationBlock(128, reduction_ratio=4)
        self.se3 = SqueezeExcitationBlock(256, reduction_ratio=4)

        # 1x1 convolution to match output channels
        self.output_conv = nn.Conv2d(256, output_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ocean encoder.

        Args:
            x: Input tensor of shape (B, C_ocean, H, W) where C_ocean=3

        Returns:
            Feature tensor of shape (B, 256, H/4, W/4)
        """
        # Initial convolution and pooling: (B, 3, H, W) -> (B, 64, H/4, W/4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1: (B, 64, H/4, W/4) -> (B, 64, H/4, W/4)
        x = self.layer1(x)
        x = self.se1(x)

        # Layer 2: (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        x = self.layer2(x)
        x = self.se2(x)

        # Layer 3: (B, 128, H/8, W/8) -> (B, 256, H/8, W/8)
        x = self.layer3(x)
        x = self.se3(x)

        # Output convolution: (B, 256, H/8, W/8) -> (B, 256, H/8, W/8)
        x = self.output_conv(x)

        return x


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for temporal processing.

    A recurrent cell that applies GRU operations with convolutional gates
    instead of fully-connected gates, suitable for spatiotemporal data.

    Args:
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        kernel_size (int): Size of convolutional kernel. Default: 3
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Reset and update gates
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=padding,
        )

        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of ConvGRU cell.

        Args:
            x: Input tensor of shape (B, C_in, H, W)
            h: Hidden state of shape (B, C_hidden, H, W)

        Returns:
            Updated hidden state of shape (B, C_hidden, H, W)
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute reset and update gates
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_channels, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # Compute candidate hidden state
        combined_candidate = torch.cat([x, reset_gate * h], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_candidate))

        # Compute new hidden state
        h_new = (1 - update_gate) * h + update_gate * candidate

        return h_new


class AtmosphereEncoder(nn.Module):
    """Atmosphere state encoder with Swin-Transformer-inspired blocks and ConvGRU.

    Encodes temporal atmosphere features (CAPE, TCWV, wind_speed) using a
    hierarchical shifted-window attention mechanism (4 stages) combined with
    ConvGRU for temporal gating (Eq. 18-19). Processes time-series input to
    256-dimensional feature map at 1/4 spatial resolution.

    Input: (B, T, C_atmo, H, W) where C_atmo=3, T=temporal steps
    Output: (B, 256, H/4, W/4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        output_channels: int = 256,
        temporal_steps: int = 4,
    ):
        """Initialize AtmosphereEncoder.

        Args:
            num_channels (int): Number of input channels. Default: 3 (CAPE, TCWV, wind)
            output_channels (int): Number of output channels. Default: 256
            temporal_steps (int): Number of temporal steps. Default: 4
        """
        super().__init__()
        self.num_channels = num_channels
        self.output_channels = output_channels
        self.temporal_steps = temporal_steps

        # Hierarchical encoder stages (simplified Swin-like structure)
        # Stage 1: (B, T, 3, H, W) -> (B, T, 64, H/2, W/2)
        self.stage1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 2: (B, T, 64, H/2, W/2) -> (B, T, 128, H/4, W/4)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Stage 3: (B, T, 128, H/4, W/4) -> (B, T, 256, H/4, W/4)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Stage 4: Residual refinement
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ConvGRU for temporal gating (Eq. 19)
        self.convgru = ConvGRUCell(256, 256, kernel_size=3)

        # Output projection
        self.output_conv = nn.Conv2d(256, output_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of atmosphere encoder with temporal processing.

        Args:
            x: Input tensor of shape (B, T, C_atmo, H, W)
                where C_atmo=3, T=temporal steps

        Returns:
            Feature tensor of shape (B, 256, H/4, W/4)
        """
        B, T, C, H, W = x.shape

        # Process each timestep through spatial stages
        features_seq = []
        for t in range(T):
            x_t = x[:, t, :, :, :]  # (B, C, H, W)

            # Hierarchical stages
            x_t = self.stage1(x_t)  # (B, 64, H/2, W/2)
            x_t = self.stage2(x_t)  # (B, 128, H/4, W/4)
            x_t = self.stage3(x_t)  # (B, 256, H/4, W/4)
            x_t = self.stage4(x_t)  # (B, 256, H/4, W/4)

            features_seq.append(x_t)

        # Temporal processing with ConvGRU (Eq. 19)
        h = torch.zeros(
            B, 256, H // 4, W // 4,
            dtype=x.dtype,
            device=x.device,
        )

        for t in range(T):
            h = self.convgru(features_seq[t], h)

        # Output projection
        output = self.output_conv(h)  # (B, 256, H/4, W/4)

        return output


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module.

    Computes cross-attention between query and key-value features with
    multiple attention heads.

    Args:
        query_dim (int): Dimension of query features
        key_dim (int): Dimension of key-value features
        num_heads (int): Number of attention heads. Default: 8
        d_per_head (int): Dimension per head. Default: 32
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        d_per_head: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_per_head = d_per_head
        self.output_dim = num_heads * d_per_head

        # Linear projections
        self.query_proj = nn.Linear(query_dim, self.output_dim)
        self.key_proj = nn.Linear(key_dim, self.output_dim)
        self.value_proj = nn.Linear(key_dim, self.output_dim)
        self.output_proj = nn.Linear(self.output_dim, query_dim)

        self.scale = math.sqrt(d_per_head)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-head cross-attention.

        Args:
            query: Tensor of shape (B, L_query, D_query)
            key: Tensor of shape (B, L_key, D_key)
            value: Tensor of shape (B, L_value, D_key)

        Returns:
            Attention output of shape (B, L_query, D_query)
        """
        B = query.shape[0]

        # Project to multiple heads
        Q = self.query_proj(query)  # (B, L_query, output_dim)
        K = self.key_proj(key)      # (B, L_key, output_dim)
        V = self.value_proj(value)  # (B, L_value, output_dim)

        # Reshape for multi-head attention
        Q = Q.view(B, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        # Now: (B, num_heads, L, d_per_head)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        # (B, num_heads, L_query, d_per_head)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, -1, self.output_dim)
        # (B, L_query, output_dim)

        # Final projection
        output = self.output_proj(context)  # (B, L_query, D_query)

        return output


class BidirectionalCrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention fusion module (Eq. 20-21).

    Performs bidirectional cross-attention between ocean and atmosphere features:
    1. Ocean attends to atmosphere (ocean as query, atmosphere as key-value)
    2. Atmosphere attends to ocean (atmosphere as query, ocean as key-value)

    Concatenates fused features and applies LayerNorm, FFN, and residual connection.

    Args:
        feature_dim (int): Dimension of input features. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        d_per_head (int): Dimension per head. Default: 32
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        d_per_head: int = 32,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = feature_dim * 2

        # Ocean -> Atmosphere attention
        self.ocean_to_atmo = MultiHeadCrossAttention(
            feature_dim, feature_dim, num_heads, d_per_head
        )

        # Atmosphere -> Ocean attention
        self.atmo_to_ocean = MultiHeadCrossAttention(
            feature_dim, feature_dim, num_heads, d_per_head
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)

        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        # Final projection for concatenated features
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        ocean_feat: torch.Tensor,
        atmo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Bidirectional cross-attention fusion (Eq. 20-21).

        Args:
            ocean_feat: Ocean features of shape (B, 256, H/4, W/4)
            atmo_feat: Atmosphere features of shape (B, 256, H/4, W/4)

        Returns:
            Fused features of shape (B, 512, H/4, W/4)
        """
        B, C, H, W = ocean_feat.shape

        # Reshape features for attention: (B, C, H, W) -> (B, HW, C)
        ocean_flat = ocean_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        atmo_flat = atmo_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Ocean attends to atmosphere
        ocean_attended = self.ocean_to_atmo(ocean_flat, atmo_flat, atmo_flat)
        ocean_fused = self.ln1(ocean_attended + ocean_flat)
        ocean_fused = ocean_fused + self.ffn1(ocean_fused)

        # Atmosphere attends to ocean
        atmo_attended = self.atmo_to_ocean(atmo_flat, ocean_flat, ocean_flat)
        atmo_fused = self.ln2(atmo_attended + atmo_flat)
        atmo_fused = atmo_fused + self.ffn2(atmo_fused)

        # Concatenate fused features
        fused = torch.cat([ocean_fused, atmo_fused], dim=-1)  # (B, HW, 512)
        fused = self.fusion_proj(fused)

        # Reshape back: (B, HW, 512) -> (B, 512, H, W)
        fused = fused.reshape(B, H, W, 512).permute(0, 3, 1, 2)

        return fused


class ClimateIndexEncoder(nn.Module):
    """Climate index encoder using MLP.

    Encodes auxiliary climate indices (EKE_index, Nino34, MJO_RMM) into
    a feature vector, then broadcasts and concatenates to spatial features.

    Args:
        num_indices (int): Number of climate indices. Default: 3 (EKE, Nino34, MJO)
        output_dim (int): Output dimension. Default: 64
    """

    def __init__(self, num_indices: int = 3, output_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_indices, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        climate_indices: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Encode climate indices and broadcast to spatial features.

        Args:
            climate_indices: Tensor of shape (B, num_indices)
            spatial_shape: Target spatial shape (H, W) for broadcasting

        Returns:
            Tensor of shape (B, output_dim, H, W) with indices broadcasted
        """
        # Encode indices
        encoded = self.mlp(climate_indices)  # (B, output_dim)

        # Reshape and broadcast
        B, output_dim = encoded.shape
        H, W = spatial_shape
        encoded = encoded.view(B, output_dim, 1, 1)
        encoded = encoded.expand(B, output_dim, H, W)

        return encoded


class PredictionHead(nn.Module):
    """Output head for rainfall field prediction.

    Progressively refines features through convolutions with batch normalization
    and ReLU activations, finally applying sigmoid for bounded output [0, 1].

    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels. Default: 1
    """

    def __init__(self, input_channels: int, output_channels: int = 1):
        super().__init__()

        self.head = nn.Sequential(
            # 1x1 convolution for channel reduction
            nn.Conv2d(input_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 3x3 convolution for spatial refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 3x3 convolution for further refinement
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final 1x1 convolution to output
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.Sigmoid(),  # Bounded output [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of prediction head.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Rainfall prediction of shape (B, 1, H, W)
        """
        return self.head(x)


class HydroAtmosFusion(nn.Module):
    """Main HydroAtmosFusion model integrating all components.

    Fuses multimodal ocean and atmosphere data using bidirectional cross-attention
    for extreme rainfall prediction. Optionally incorporates auxiliary climate indices.

    Architecture:
    1. OceanEncoder: 3-channel ocean input -> 256-dim features
    2. AtmosphereEncoder: T-step 3-channel atmosphere input -> 256-dim features
    3. ClimateIndexEncoder: Climate indices -> spatial features (optional)
    4. BidirectionalCrossModalAttention: Fuse ocean + atmosphere -> 512-dim features
    5. PredictionHead: Output rainfall field (B, 1, H, W)
    """

    def __init__(
        self,
        ocean_channels: int = 3,
        atmo_channels: int = 3,
        temporal_steps: int = 4,
        num_climate_indices: int = 3,
        feature_dim: int = 256,
        num_attention_heads: int = 8,
        attention_d_per_head: int = 32,
        use_climate_indices: bool = True,
    ):
        """Initialize HydroAtmosFusion model.

        Args:
            ocean_channels (int): Number of ocean input channels. Default: 3
            atmo_channels (int): Number of atmosphere input channels. Default: 3
            temporal_steps (int): Number of temporal atmosphere steps. Default: 4
            num_climate_indices (int): Number of climate indices. Default: 3
            feature_dim (int): Dimension of intermediate features. Default: 256
            num_attention_heads (int): Number of attention heads. Default: 8
            attention_d_per_head (int): Dimension per attention head. Default: 32
            use_climate_indices (bool): Whether to use climate indices. Default: True
        """
        super().__init__()

        self.ocean_channels = ocean_channels
        self.atmo_channels = atmo_channels
        self.temporal_steps = temporal_steps
        self.use_climate_indices = use_climate_indices

        # Encoders
        self.ocean_encoder = OceanEncoder(
            num_channels=ocean_channels,
            output_channels=feature_dim,
        )

        self.atmo_encoder = AtmosphereEncoder(
            num_channels=atmo_channels,
            output_channels=feature_dim,
            temporal_steps=temporal_steps,
        )

        # Climate index encoder
        if use_climate_indices:
            self.climate_encoder = ClimateIndexEncoder(
                num_indices=num_climate_indices,
                output_dim=64,
            )

        # Cross-modal attention
        self.cross_attention = BidirectionalCrossModalAttention(
            feature_dim=feature_dim,
            num_heads=num_attention_heads,
            d_per_head=attention_d_per_head,
        )

        # Prediction head
        fused_dim = feature_dim * 2  # 512
        if use_climate_indices:
            fused_dim += 64  # Add climate index features

        self.pred_head = PredictionHead(
            input_channels=fused_dim,
            output_channels=1,
        )

    def forward(
        self,
        ocean: torch.Tensor,
        atmosphere: torch.Tensor,
        climate_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of HydroAtmosFusion.

        Args:
            ocean: Ocean features of shape (B, 3, H, W)
                Contains SST, SSH, EKE
            atmosphere: Atmosphere features of shape (B, T, 3, H, W)
                Contains CAPE, TCWV, wind_speed over T timesteps
            climate_indices: Optional climate indices of shape (B, num_indices)
                Contains auxiliary climate information (e.g., Nino34, MJO)

        Returns:
            Rainfall prediction of shape (B, 1, H, W) with values in [0, 1]
        """
        # Encode ocean features
        ocean_feat = self.ocean_encoder(ocean)  # (B, 256, H/4, W/4)

        # Encode atmosphere features
        atmo_feat = self.atmo_encoder(atmosphere)  # (B, 256, H/4, W/4)

        # Bidirectional cross-attention fusion
        fused = self.cross_attention(ocean_feat, atmo_feat)  # (B, 512, H/4, W/4)

        # Optional: Include climate indices
        if self.use_climate_indices and climate_indices is not None:
            climate_feat = self.climate_encoder(
                climate_indices,
                spatial_shape=(fused.shape[2], fused.shape[3]),
            )  # (B, 64, H/4, W/4)
            fused = torch.cat([fused, climate_feat], dim=1)  # (B, 576, H/4, W/4)

        # Predict rainfall field
        rainfall = self.pred_head(fused)  # (B, 1, H/4, W/4)

        return rainfall


if __name__ == "__main__":
    # Example usage
    B, T, H, W = 4, 4, 64, 64

    # Create model
    model = HydroAtmosFusion(
        ocean_channels=3,
        atmo_channels=3,
        temporal_steps=T,
        num_climate_indices=3,
        feature_dim=256,
        num_attention_heads=8,
        attention_d_per_head=32,
        use_climate_indices=True,
    )

    # Create dummy inputs
    ocean_input = torch.randn(B, 3, H, W)
    atmo_input = torch.randn(B, T, 3, H, W)
    climate_input = torch.randn(B, 3)

    # Forward pass
    rainfall_pred = model(ocean_input, atmo_input, climate_input)

    print(f"Ocean input shape: {ocean_input.shape}")
    print(f"Atmosphere input shape: {atmo_input.shape}")
    print(f"Climate indices shape: {climate_input.shape}")
    print(f"Rainfall output shape: {rainfall_pred.shape}")
    print(f"Output range: [{rainfall_pred.min():.4f}, {rainfall_pred.max():.4f}]")
