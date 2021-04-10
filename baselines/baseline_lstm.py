"""
Baseline LSTM.

Contains the definition of the BaselineLSTM class.

Takes in a sequence of token IDs as input and makes a single
prediction about the sequence at the end.

Trained on symbolic data in raw sequence form using cross-entropy loss.

The implementation in the paper uses 50-dimensional embeddings and
50-dimensional hidden layers, predicting a single output, which is
treated as in a classification problem.
"""

import torch
import torch.nn as nn
# noinspection PyPep8Naming
from torch.nn import functional as F


class BaselineLSTM(nn.Module):
    """Baseline LSTM class."""

    def __init__(self, input_size, hidden_size, n_targets=1, output_size=None,
                 embedding_size=50, use_embeddings=False):
        super(BaselineLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else self.input_size
        self.n_targets = n_targets + (
            1 if n_targets > 1 else 0)  # special eos symbol

        if use_embeddings:
            self.input_size = embedding_size
            self.embedding = nn.Embedding(num_embeddings=self.input_size + 1,
                                          embedding_dim=embedding_size)
        else:
            self.embedding = nn.Identity()

        self.lstm_layer = nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  proj_size=self.output_size, batch_first=True)

    def forward(self, sequences):
        """
        Args:
            sequences: [batch, seq_len, input_size]
        Returns:
            output: [batch, n_outputs, output_size]; output_size=max_token_id
            hidden: [batch, num_layers=1, hidden_size]
        """
        sequences = self.embedding(sequences)
        output, hidden = self.lstm_layer(sequences)
        return output[:, -self.n_targets:, :], hidden

    def reset_parameters(self):
        self.lstm_layer.reset_parameters()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
