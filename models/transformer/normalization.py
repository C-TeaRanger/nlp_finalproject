"""
Normalization layers for Transformer
Supports LayerNorm and RMSNorm
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    More efficient alternative to LayerNorm
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_normalized = x / rms

        # Scale
        return self.weight * x_normalized


def get_norm_layer(norm_type: str, d_model: int, eps: float = 1e-6):
    """Factory function to get normalization layer"""
    if norm_type == 'LayerNorm':
        return nn.LayerNorm(d_model, eps=eps)
    elif norm_type == 'RMSNorm':
        return RMSNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


if __name__ == "__main__":
    # Test normalization layers
    batch_size = 4
    seq_len = 10
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model)

    print("Testing Normalization Layers\n")

    for norm_type in ['LayerNorm', 'RMSNorm']:
        print(f"{norm_type}:")
        norm = get_norm_layer(norm_type, d_model)

        output = norm(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Input mean: {x.mean():.4f}, std: {x.std():.4f}")
        print(f"  Output mean: {output.mean():.4f}, std: {output.std():.4f}")
        print(f"  Parameters: {sum(p.numel() for p in norm.parameters()):,}")
        print()
