"""
Attention mechanisms for RNN-based NMT
Implements dot-product, multiplicative, and additive attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    """Dot-product attention: score(h_t, h_s) = h_t^T h_s"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len)

        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, src_len)
        """
        # Compute attention scores
        # (batch_size, hidden_dim, 1)
        decoder_hidden = decoder_hidden.unsqueeze(2)

        d_k = encoder_outputs.size(-1) # 512

        # (batch_size, src_len, hidden_dim) @ (batch_size, hidden_dim, 1)
        # -> (batch_size, src_len, 1)
        scores = torch.bmm(encoder_outputs, decoder_hidden)
        scores = scores.squeeze(2) / (d_k ** 0.5)  # (batch_size, src_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e10)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, src_len)

        # Compute context vector
        # (batch_size, 1, src_len) @ (batch_size, src_len, hidden_dim)
        # -> (batch_size, 1, hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch_size, hidden_dim)

        return context, attention_weights


class MultiplicativeAttention(nn.Module):
    """Multiplicative attention: score(h_t, h_s) = h_t^T W h_s"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len)

        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, src_len)
        """
        # Transform encoder outputs
        # (batch_size, src_len, hidden_dim)
        transformed_encoder = self.W(encoder_outputs)

        # Compute attention scores
        # (batch_size, hidden_dim, 1)
        decoder_hidden = decoder_hidden.unsqueeze(2)

        # (batch_size, src_len, hidden_dim) @ (batch_size, hidden_dim, 1)
        # -> (batch_size, src_len, 1)
        scores = torch.bmm(transformed_encoder, decoder_hidden)
        scores = scores.squeeze(2) # (batch_size, src_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e10)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, src_len)

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch_size, hidden_dim)

        return context, attention_weights


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention: score(h_t, h_s) = v^T tanh(W_1 h_t + W_2 h_s)"""

    def __init__(self, hidden_dim: int, attention_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim

        self.W_decoder = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.W_encoder = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len)

        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, src_len)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        # Transform decoder hidden state
        # (batch_size, attention_dim)
        decoder_transform = self.W_decoder(decoder_hidden)
        # Expand to match encoder outputs
        # (batch_size, src_len, attention_dim)
        decoder_transform = decoder_transform.unsqueeze(1).expand(-1, src_len, -1)
        # Transform encoder outputs
        # (batch_size, src_len, attention_dim)
        encoder_transform = self.W_encoder(encoder_outputs)
        # Compute attention scores
        # (batch_size, src_len, attention_dim)
        combined = torch.tanh(decoder_transform + encoder_transform)
        # (batch_size, src_len, 1) -> (batch_size, src_len)
        scores = self.v(combined).squeeze(2)
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e10)
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, src_len)
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        return context, attention_weights


def get_attention(attention_type: str, hidden_dim: int):
    """Factory function to get attention mechanism"""
    if attention_type == 'dot':
        return DotProductAttention(hidden_dim)
    elif attention_type == 'multiplicative':
        return MultiplicativeAttention(hidden_dim)
    elif attention_type == 'additive':
        return AdditiveAttention(hidden_dim)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    # Test attention mechanisms
    batch_size = 4
    src_len = 10
    hidden_dim = 512

    decoder_hidden = torch.randn(batch_size, hidden_dim)
    encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)

    print("Testing attention mechanisms\n")

    for attn_type in ['dot', 'multiplicative', 'additive']:
        print(f"{attn_type.capitalize()} Attention:")
        attention = get_attention(attn_type, hidden_dim)

        context, weights = attention(decoder_hidden, encoder_outputs)
        print(f"  Context shape: {context.shape}")
        print(f"  Attention weights shape: {weights.shape}")
        print(f"  Attention weights sum: {weights.sum(dim=1)}")
        print()
