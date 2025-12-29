"""
Transformer Encoder layer
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .normalization import get_norm_layer


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer"""

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm_type: str = 'LayerNorm',
                 use_relative_pos: bool = False):
        super().__init__()

        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative_pos)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Normalization layers
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, relative_bias=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len)
            relative_bias: (seq_len, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask, relative_bias)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder (stack of encoder layers)"""

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm_type: str = 'LayerNorm',
                 pad_idx: int = 0,
                 use_relative_pos: bool = False):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Scale embedding
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, norm_type, use_relative_pos)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, relative_bias=None):
        """
        Args:
            src: (batch_size, src_len)
            src_mask: (batch_size, 1, 1, src_len)
            relative_bias: (src_len, src_len, d_model)

        Returns:
            (batch_size, src_len, d_model)
        """
        # Embedding
        x = self.embedding(src) * self.scale
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask, relative_bias)

        return x


if __name__ == "__main__":
    # Test encoder
    batch_size = 4
    src_len = 10
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048

    print("Testing Transformer Encoder\n")

    for norm_type in ['LayerNorm', 'RMSNorm']:
        print(f"Encoder with {norm_type}:")

        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            norm_type=norm_type
        )

        # Create dummy input
        src = torch.randint(1, vocab_size, (batch_size, src_len))

        # Forward pass
        output = encoder(src)

        print(f"  Input shape: {src.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print()
