"""
RNN Encoder for machine translation
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    """
    RNN Encoder with 2 unidirectional layers
    Supports both LSTM and GRU
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 cell_type: str = 'LSTM',
                 pad_idx: int = 0):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Dropout
        self.embed_dropout = nn.Dropout(dropout)

        # RNN layer
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    def forward(self, src, src_lengths):
        """
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)

        Returns:
            outputs: (batch_size, src_len, hidden_dim)
            hidden: tuple of (num_layers, batch_size, hidden_dim) for LSTM
                    or (num_layers, batch_size, hidden_dim) for GRU
        """
        # Embedding
        embedded = self.embedding(src)  # (batch_size, src_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        # Pack padded sequence for efficient RNN processing
        packed = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # RNN forward
        packed_outputs, hidden = self.rnn(packed)

        # Unpack sequence
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: (batch_size, src_len, hidden_dim)

        return outputs, hidden


if __name__ == "__main__":
    # Test encoder
    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512
    batch_size = 4
    src_len = 10

    print("Testing RNN Encoder\n")

    for cell_type in ['LSTM', 'GRU']:
        print(f"{cell_type} Encoder:")
        encoder = RNNEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.3,
            cell_type=cell_type
        )

        # Create dummy input
        src = torch.randint(0, vocab_size, (batch_size, src_len))
        src_lengths = torch.tensor([10, 8, 6, 5])

        # Forward pass
        outputs, hidden = encoder(src, src_lengths)

        print(f"  Input shape: {src.shape}")
        print(f"  Output shape: {outputs.shape}")
        if cell_type == 'LSTM':
            print(f"  Hidden state shape: h={hidden[0].shape}, c={hidden[1].shape}")
        else:
            print(f"  Hidden state shape: {hidden.shape}")
        print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print()
