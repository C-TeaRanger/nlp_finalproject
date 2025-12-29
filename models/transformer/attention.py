"""
Multi-head attention mechanism for Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optional relative position bias"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_relative_pos: bool = False):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_relative_pos = use_relative_pos

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Relative position projection (if using relative positions)
        if use_relative_pos:
            self.W_r = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.d_k)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k)
        Transpose to get: (batch_size, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None, relative_bias=None):
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, value_len, d_model)
            mask: (batch_size, 1, 1, key_len) or (batch_size, 1, query_len, key_len)
            relative_bias: (query_len, key_len, d_model) - relative position bias

        Returns:
            output: (batch_size, query_len, d_model)
            attention_weights: (batch_size, num_heads, query_len, key_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch_size, query_len, d_model)
        K = self.W_k(key)    # (batch_size, key_len, d_model)
        V = self.W_v(value)  # (batch_size, value_len, d_model)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, query_len, d_k)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, key_len, d_k)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, value_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Add relative position bias if provided
        if self.use_relative_pos and relative_bias is not None:
            # Project relative bias: (query_len, key_len, d_model) -> (query_len, key_len, d_model)
            R = self.W_r(relative_bias)  # (query_len, key_len, d_model)

            # Split into heads: (query_len, key_len, num_heads, d_k)
            R = R.view(R.size(0), R.size(1), self.num_heads, self.d_k)
            # Transpose to (num_heads, query_len, key_len, d_k)
            R = R.permute(2, 0, 1, 3)

            # Compute relative position scores using einsum
            # Q: (batch_size, num_heads, query_len, d_k)
            # R: (num_heads, query_len, key_len, d_k)
            # Output: (batch_size, num_heads, query_len, key_len)
            rel_scores = torch.einsum('bnqd,nqkd->bnqk', Q, R) / self.scale

            # Add to attention scores
            scores = scores + rel_scores

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e10)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights


if __name__ == "__main__":
    # Test multi-head attention
    batch_size = 4
    seq_len = 10
    d_model = 512
    num_heads = 8

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    print("Testing Multi-Head Attention\n")

    attention = MultiHeadAttention(d_model, num_heads)

    output, attn_weights = attention(query, key, value)

    print(f"Input shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")
    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attn_weights.shape}")
    print(f"\nParameters: {sum(p.numel() for p in attention.parameters()):,}")

    # Test with mask
    print("\n\nTesting with causal mask:")
    mask = torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1).bool()
    output_masked, attn_weights_masked = attention(query, key, value, mask)
    print(f"Output shape: {output_masked.shape}")
    print(f"Attention weights shape: {attn_weights_masked.shape}")
