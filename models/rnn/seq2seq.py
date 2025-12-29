"""
Sequence-to-Sequence model with attention
"""

import torch
import torch.nn as nn
import random
from .encoder import RNNEncoder
from .decoder import RNNDecoder


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining encoder and decoder
    """

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 cell_type: str = 'LSTM',
                 attention_type: str = 'dot',
                 pad_idx: int = 0):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.attention_type = attention_type
        self.pad_idx = pad_idx

        # Encoder
        self.encoder = RNNEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
            pad_idx=pad_idx
        )

        # Decoder
        self.decoder = RNNDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
            attention_type=attention_type,
            pad_idx=pad_idx
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=1.0):
        """
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: (batch_size, tgt_len, tgt_vocab_size)
            attention_weights: List of (batch_size, src_len) for each time step
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.tgt_vocab_size

        # Encode source
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        # encoder_outputs: (batch_size, src_len, hidden_dim)

        # Create padding mask for attention
        src_mask = (src == self.pad_idx)  # (batch_size, src_len)

        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)

        # Store attention weights
        attention_weights_list = []

        # First decoder input is <bos> token
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)
        # breakpoint()

        for t in range(1, tgt_len):
            # Decode one step
            output, hidden, attention_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            # output: (batch_size, 1, hidden_dim)
            # attention_weights: (batch_size, src_len)

            # Project to vocabulary
            logits = self.fc_out(output)  # (batch_size, 1, tgt_vocab_size)
            outputs[:, t, :] = logits.squeeze(1)

            # Store attention weights
            attention_weights_list.append(attention_weights)

            # Teacher forcing: use ground truth as next input
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                # Use model's own prediction
                decoder_input = logits.argmax(dim=-1)  # (batch_size, 1)

        return outputs, attention_weights_list

    # def encode(self, src, src_lengths):
    #     """Encode source sequence"""
    #     return self.encoder(src, src_lengths)

    def decode_step(self, tgt, hidden, encoder_outputs, src_mask=None):
        """Single decoding step"""
        output, hidden, attention_weights = self.decoder(tgt, hidden, encoder_outputs, src_mask)
        logits = self.fc_out(output)
        return logits, hidden, attention_weights


if __name__ == "__main__":
    # Test Seq2Seq model
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    embed_dim = 256
    hidden_dim = 512
    batch_size = 4
    src_len = 10
    tgt_len = 12

    print("Testing Seq2Seq Model\n")

    for cell_type in ['LSTM', 'GRU']:
        for attention_type in ['dot', 'multiplicative', 'additive']:
            print(f"{cell_type} Seq2Seq with {attention_type} attention:")

            model = Seq2Seq(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=0.3,
                cell_type=cell_type,
                attention_type=attention_type
            )

            # Create dummy inputs
            src = torch.randint(1, src_vocab_size, (batch_size, src_len))
            src_lengths = torch.tensor([10, 9, 8, 7])
            tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

            # Forward pass
            outputs, attn_weights = model(src, src_lengths, tgt, teacher_forcing_ratio=1.0)

            print(f"  Source shape: {src.shape}")
            print(f"  Target shape: {tgt.shape}")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Number of attention weight tensors: {len(attn_weights)}")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print()

            # Test with teacher forcing = 0.5
            outputs, attn_weights = model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)
            print(f"  With teacher_forcing_ratio=0.5: Output shape: {outputs.shape}")
            print()
