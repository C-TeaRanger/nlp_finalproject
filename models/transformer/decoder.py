"""
Transformer Decoder layer
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .normalization import get_norm_layer


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer"""

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm_type: str = 'LayerNorm',
                 use_relative_pos: bool = False):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative_pos)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, False)

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
        self.norm3 = get_norm_layer(norm_type, d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, tgt_relative_bias=None):
        """
        Args:
            x: (batch_size, tgt_len, d_model)
            encoder_output: (batch_size, src_len, d_model)
            src_mask: (batch_size, 1, 1, src_len) - encoder padding mask
            tgt_mask: (tgt_len, tgt_len) - decoder causal mask
            tgt_relative_bias: (tgt_len, tgt_len, d_model) - relative position bias for self-attention

        Returns:
            (batch_size, tgt_len, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, tgt_mask, tgt_relative_bias)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention with residual connection
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder (stack of decoder layers)"""

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

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, norm_type, use_relative_pos)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None, tgt_relative_bias=None):
        """
        Args:
            tgt: (batch_size, tgt_len)
            encoder_output: (batch_size, src_len, d_model)
            src_mask: (batch_size, 1, 1, src_len)
            tgt_mask: (tgt_len, tgt_len)
            tgt_relative_bias: (tgt_len, tgt_len, d_model)

        Returns:
            (batch_size, tgt_len, d_model)
        """
        # Embedding
        x = self.embedding(tgt) * self.scale
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask, tgt_relative_bias)

        return x


if __name__ == "__main__":
    # Test decoder
    batch_size = 4
    src_len = 10
    tgt_len = 12
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048

    print("Testing Transformer Decoder\n")

    for norm_type in ['LayerNorm', 'RMSNorm']:
        print(f"Decoder with {norm_type}:")

        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            norm_type=norm_type
        )

        # Create dummy inputs
        tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # Create causal mask
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()

        # Forward pass
        output = decoder(tgt, encoder_output, tgt_mask=tgt_mask)

        print(f"  Target shape: {tgt.shape}")
        print(f"  Encoder output shape: {encoder_output.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
        print()
