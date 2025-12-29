"""
Complete Transformer model for machine translation
"""

import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .positional_encoding import get_positional_encoding


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence translation
    """

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 position_embedding: str = 'sinusoidal',
                 norm_type: str = 'LayerNorm',
                 max_len: int = 5000,
                 pad_idx: int = 0):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.use_relative_pos = (position_embedding == 'relative')

        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_type=norm_type,
            pad_idx=pad_idx,
            use_relative_pos=self.use_relative_pos
        )

        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_type=norm_type,
            pad_idx=pad_idx,
            use_relative_pos=self.use_relative_pos
        )

        # Positional encoding
        self.src_pos_encoding = get_positional_encoding(position_embedding, d_model, max_len, dropout)
        self.tgt_pos_encoding = get_positional_encoding(position_embedding, d_model, max_len, dropout)

        # Output projection
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        Create mask for source sequence (padding mask)

        Args:
            src: (batch_size, src_len)

        Returns:
            (batch_size, 1, 1, src_len)
        """
        src_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        Create mask for target sequence (causal mask + padding mask)

        Args:
            tgt: (batch_size, tgt_len)

        Returns:
            (batch_size, 1, tgt_len, tgt_len)
        """
        batch_size, tgt_len = tgt.size()

        # Padding mask
        tgt_padding_mask = (tgt == self.pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, tgt_len)

        # Causal mask
        tgt_causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device),
            diagonal=1
        ).bool()
        # (tgt_len, tgt_len)

        # Combine masks
        tgt_mask = tgt_padding_mask | tgt_causal_mask
        # (batch_size, 1, tgt_len, tgt_len)

        return tgt_mask

    def encode(self, src, src_mask=None):
        """
        Encode source sequence

        Args:
            src: (batch_size, src_len)
            src_mask: (batch_size, 1, 1, src_len)

        Returns:
            (batch_size, src_len, d_model)
        """
        # Embedding + positional encoding
        src_embedded = self.encoder.embedding(src) * self.encoder.scale
        src_embedded = self.src_pos_encoding(src_embedded)

        # Compute relative position bias if using relative positional encoding
        src_len = src.size(1)
        src_relative_bias = None
        if self.use_relative_pos:
            src_relative_bias = self.src_pos_encoding.get_relative_bias(src_len, src.device)

        # Encode
        encoder_output = src_embedded
        for layer in self.encoder.layers:
            encoder_output = layer(encoder_output, src_mask, src_relative_bias)

        return encoder_output

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence

        Args:
            tgt: (batch_size, tgt_len)
            encoder_output: (batch_size, src_len, d_model)
            src_mask: (batch_size, 1, 1, src_len)
            tgt_mask: (batch_size, 1, tgt_len, tgt_len)

        Returns:
            (batch_size, tgt_len, d_model)
        """
        # Embedding + positional encoding
        tgt_embedded = self.decoder.embedding(tgt) * self.decoder.scale
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)

        # Compute relative position bias if using relative positional encoding
        tgt_len = tgt.size(1)
        tgt_relative_bias = None
        if self.use_relative_pos:
            tgt_relative_bias = self.tgt_pos_encoding.get_relative_bias(tgt_len, tgt.device)

        # Decode
        decoder_output = tgt_embedded
        for layer in self.decoder.layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask, tgt_relative_bias)

        return decoder_output

    def forward(self, src, tgt):
        """
        Forward pass

        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)

        Returns:
            (batch_size, tgt_len, tgt_vocab_size)
        """
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encode
        encoder_output = self.encode(src, src_mask)

        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.generator(decoder_output)

        return output


if __name__ == "__main__":
    # Test Transformer model
    batch_size = 4
    src_len = 10
    tgt_len = 12
    src_vocab_size = 10000
    tgt_vocab_size = 8000

    print("Testing Transformer Model\n")

    configs = [
        {'position_embedding': 'sinusoidal', 'norm_type': 'LayerNorm'},
        {'position_embedding': 'learned', 'norm_type': 'LayerNorm'},
        {'position_embedding': 'sinusoidal', 'norm_type': 'RMSNorm'},
    ]

    for config in configs:
        print(f"Configuration: {config}")

        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            **config
        )

        # Create dummy inputs
        src = torch.randint(1, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

        # Forward pass
        output = model(src, tgt)

        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
