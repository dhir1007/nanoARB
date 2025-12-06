"""Mamba-based LOB prediction model for sub-microsecond inference."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MambaBlock(nn.Module):
    """Mamba block for efficient sequence modeling.

    Uses selective state space models for O(L) complexity instead of O(L^2) for attention.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        # x_proj outputs: dt_rank (for delta projection) + 2*d_state (for B and C)
        dt_rank = max(d_state // 2, 1)
        self.dt_rank = dt_rank
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize A (diagonal state matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, "b d l -> b l d")

        # Activation
        x = F.silu(x)

        # SSM
        y = self.ssm(x)

        # Gate and output
        y = y * F.silu(z)
        output = self.out_proj(y)

        return self.dropout(output)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state space model computation."""
        batch, seq_len, _ = x.shape
        device = x.device

        # Get SSM parameters
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # Project x to get dt_proj_input, B, C
        x_proj = self.x_proj(x)
        dt_proj_input, B, C = x_proj.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_proj_input))

        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)

        # Selective scan
        y = self.selective_scan(x, deltaA, deltaB, C, D)

        return y

    def selective_scan(
        self,
        x: torch.Tensor,
        deltaA: torch.Tensor,
        deltaB: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan operation."""
        batch, seq_len, d_inner = x.shape

        # Initialize state: [batch, d_inner, d_state]
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # x_t: [batch, d_inner]
            x_t = x[:, t, :]
            
            # Update state: h = A * h + B * x
            # deltaA[:, t]: [batch, d_inner, d_state]
            # deltaB[:, t]: [batch, d_inner, d_state]
            # x_t.unsqueeze(-1): [batch, d_inner, 1]
            h = deltaA[:, t] * h + deltaB[:, t] * x_t.unsqueeze(-1)
            
            # Output: y = C * h + D * x
            # C[:, t]: [batch, d_state]
            # h: [batch, d_inner, d_state]
            # (h * C[:, t].unsqueeze(1)): [batch, d_inner, d_state]
            # sum over d_state: [batch, d_inner]
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1) + D * x_t
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class MambaLOBModel(nn.Module):
    """Mamba-based model for LOB price prediction.

    Designed for sub-microsecond inference latency when exported to ONNX.
    """

    def __init__(
        self,
        input_dim: int = 40,  # 10 levels * 4 features
        hidden_dim: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        num_horizons: int = 3,
        num_classes: int = 3,  # down, neutral, up
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        self.num_classes = num_classes

        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output heads (one per horizon)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            for _ in range(num_horizons)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_features: If True, also return intermediate features

        Returns:
            Logits tensor of shape (batch, num_horizons, num_classes)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Mamba layers with residual connections
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        # Take last timestep
        features = x[:, -1, :]

        # Output heads
        outputs = []
        for head in self.output_heads:
            outputs.append(head(features))

        logits = torch.stack(outputs, dim=1)

        if return_features:
            return logits, features
        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with confidence scores.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predictions, confidences) both of shape (batch, num_horizons)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1) - 1  # Map to -1, 0, 1
        confidences = probs.max(dim=-1).values
        return predictions, confidences


class TransactionCostAwareLoss(nn.Module):
    """Loss function that accounts for transaction costs."""

    def __init__(
        self,
        spread_penalty: float = 0.001,
        slippage_estimate: float = 0.0005,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.spread_penalty = spread_penalty
        self.slippage_estimate = slippage_estimate
        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        magnitudes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            logits: Model outputs (batch, num_horizons, num_classes)
            targets: Target classes (batch, num_horizons)
            magnitudes: Optional price move magnitudes for weighting

        Returns:
            Scalar loss value
        """
        batch, num_horizons, num_classes = logits.shape

        # Flatten for cross entropy
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)

        # Base cross entropy loss
        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=self.class_weights,
            reduction='none'
        ).view(batch, num_horizons)

        # Add transaction cost penalty for predictions that change direction
        probs = F.softmax(logits, dim=-1)
        # Penalize uncertain predictions (low confidence)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        cost_penalty = self.spread_penalty * entropy

        # Weight by magnitude if provided
        if magnitudes is not None:
            # Higher magnitude moves are more important
            importance = 1 + magnitudes.abs()
            ce_loss = ce_loss * importance

        total_loss = ce_loss.mean() + cost_penalty.mean()

        return total_loss


def export_to_onnx(
    model: MambaLOBModel,
    output_path: str,
    sequence_length: int = 100,
    opset_version: int = 17,
) -> None:
    """Export model to ONNX format for Rust inference.

    Args:
        model: Trained model
        output_path: Path to save ONNX model
        sequence_length: Input sequence length
        opset_version: ONNX opset version
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, model.input_dim)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    # Test model creation and forward pass
    model = MambaLOBModel(
        input_dim=40,
        hidden_dim=128,
        num_layers=4,
        num_horizons=3,
    )

    # Test input
    x = torch.randn(2, 100, 40)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test prediction
    preds, confs = model.predict(x)
    print(f"Predictions shape: {preds.shape}")
    print(f"Confidences shape: {confs.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

