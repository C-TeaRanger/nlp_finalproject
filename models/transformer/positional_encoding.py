"""
Positional Encoding for Transformer
Supports sinusoidal, learned, and relative positional encodings
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (original Transformer)"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding (Shaw et al. 2018)
    Encodes relative distances between positions for use in attention
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Relative position embeddings: from -(max_len-1) to +(max_len-1)
        # Index 0 corresponds to relative distance -(max_len-1)
        # Index max_len-1 corresponds to relative distance 0
        # Index 2*max_len-2 corresponds to relative distance +(max_len-1)
        self.relative_pe = nn.Parameter(torch.randn(2 * max_len - 1, d_model))
        nn.init.xavier_uniform_(self.relative_pe)

    def forward(self, x):
        """
        For interface compatibility with other positional encodings.
        Relative encoding doesn't modify embeddings directly.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            x unchanged: (batch_size, seq_len, d_model)
        """
        return x

    def get_relative_bias(self, seq_len: int, device):
        """
        Get relative position bias matrix for attention

        Args:
            seq_len: sequence length
            device: device to create tensor on

        Returns:
            relative_bias: (seq_len, seq_len, d_model)
        """
        # Compute relative distances: (i, j) -> j - i
        # Shape: (seq_len, seq_len)
        positions = torch.arange(seq_len, device=device)
        rel_distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)

        # Clamp to valid range and convert to indices
        # Clamp to [-max_len+1, max_len-1] then shift to [0, 2*max_len-2]
        rel_distances = torch.clamp(rel_distances, -self.max_len + 1, self.max_len - 1)
        indices = rel_distances + self.max_len - 1  # (seq_len, seq_len)

        # Get embeddings: (seq_len, seq_len, d_model)
        relative_bias = self.relative_pe[indices]

        return relative_bias


def get_positional_encoding(pos_type: str, d_model: int, max_len: int = 5000, dropout: float = 0.1):
    """Factory function to get positional encoding"""
    if pos_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)
    elif pos_type == 'learned':
        return LearnedPositionalEncoding(d_model, max_len, dropout)
    elif pos_type == 'relative':
        return RelativePositionalEncoding(d_model, max_len, dropout)
    else:
        raise ValueError(f"Unknown positional encoding type: {pos_type}")


if __name__ == "__main__":
    # Test positional encodings
    batch_size = 4
    seq_len = 10
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model)

    print("Testing Positional Encodings\n")

    for pos_type in ['sinusoidal', 'learned', 'relative']:
        print(f"{pos_type.capitalize()} Positional Encoding:")
        pe = get_positional_encoding(pos_type, d_model)

        output = pe(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in pe.parameters()):,}")
        print()
