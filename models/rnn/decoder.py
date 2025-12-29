"""
RNN Decoder with attention for machine translation
"""

import torch
import torch.nn as nn
from .attention import get_attention


class RNNDecoder(nn.Module):
    """
    RNN Decoder with attention mechanism
    2 unidirectional layers
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 cell_type: str = 'LSTM',
                 attention_type: str = 'dot',
                 pad_idx: int = 0):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.attention_type = attention_type
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Dropout
        self.embed_dropout = nn.Dropout(dropout)

        # RNN layer
        # Input: embedding + context vector
        rnn_input_dim = embed_dim + hidden_dim

        if cell_type == 'LSTM':
            # self.rnn = nn.LSTM(
            #     rnn_input_dim,
            #     hidden_dim,
            #     num_layers=num_layers,
            #     batch_first=True,
            #     dropout=dropout if num_layers > 1 else 0
            # )
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.W_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                rnn_input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        # Attention mechanism
        self.attention = get_attention(attention_type, hidden_dim)

        # Project [RNN_Output; Context] -> [Hidden_Dim]
        # self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Output dropout
        # self.output_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, hidden, encoder_outputs, mask=None):
        """
        Args:
            tgt: (batch_size, 1) - single token input
            hidden: Previous hidden state
                    For LSTM: (h, c) where h, c are (num_layers, batch_size, hidden_dim)
                    For GRU: (num_layers, batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len) - padding mask for encoder outputs

        Returns:
            output: (batch_size, 1, hidden_dim)
            hidden: Updated hidden state
            attention_weights: (batch_size, src_len)
        """
        # Embedding
        embedded = self.embedding(tgt)  # (batch_size, 1, embed_dim)
        embedded = self.embed_dropout(embedded)

        rnn_output, hidden = self.rnn(embedded, hidden)
        # Get last layer hidden state for attention
        if self.cell_type == 'LSTM':
            last_hidden = hidden[0][-1]  # (batch_size, hidden_dim)
        else:
            last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Compute attention context
        context, attention_weights = self.attention(last_hidden, encoder_outputs, mask)
        # context: (batch_size, hidden_dim)
        # attention_weights: (batch_size, src_len)

        # Concatenate embedding and context
        # (batch_size, 1, embed_dim + hidden_dim)
        # rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output_combined = torch.cat([rnn_output, context.unsqueeze(1)], dim=2)
        output_combined = self.dropout(output_combined)

        # RNN forward
        # output, hidden = self.rnn(rnn_input, hidden)
        # output: (batch_size, 1, hidden_dim)
        output = torch.tanh(self.W_combine(output_combined))

        # # 1. Concatenate the RNN output with the Context Vector
        # # context: (batch, hidden) -> unsqueeze -> (batch, 1, hidden)
        # combined_output = torch.cat((output, context.unsqueeze(1)), dim=2) 
        # # combined_output shape: (batch, 1, hidden_dim * 2)

        # # 2. Project back to hidden_dim using the new layer
        # projected_output = torch.tanh(self.concat(combined_output))
        # # projected_output shape: (batch, 1, hidden_dim)

        # # 3. Return the PROJECTED output, not the raw RNN output
        # output = self.output_dropout(projected_output)

        return output, hidden, attention_weights


if __name__ == "__main__":
    # Test decoder
    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512
    batch_size = 4
    src_len = 10

    print("Testing RNN Decoder with different attention mechanisms\n")

    for cell_type in ['LSTM', 'GRU']:
        for attention_type in ['dot', 'multiplicative', 'additive']:
            print(f"{cell_type} Decoder with {attention_type} attention:")

            decoder = RNNDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=0.3,
                cell_type=cell_type,
                attention_type=attention_type
            )

            # Create dummy inputs
            tgt = torch.randint(0, vocab_size, (batch_size, 1))
            encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)

            if cell_type == 'LSTM':
                h = torch.randn(2, batch_size, hidden_dim)
                c = torch.randn(2, batch_size, hidden_dim)
                hidden = (h, c)
            else:
                hidden = torch.randn(2, batch_size, hidden_dim)

            # Forward pass
            output, new_hidden, attn_weights = decoder(tgt, hidden, encoder_outputs)

            print(f"  Input shape: {tgt.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Attention weights shape: {attn_weights.shape}")
            print(f"  Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
            print()
