"""
Baseline RNN.

Contains the definition of the BaselineRNN class.

Takes in a sequence of token IDs as input and makes a single 
prediction about the sequence at the end.

Trained on symbolic data in raw sequence form using cross-entropy loss.

The implementation in the paper uses 50-dimensional embeddings and
50-dimensional hidden layers, predicting a single output, which is
treated as in a classification problem.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BaselineRNN(nn.Module):
    """Baseline RNN class.

    Implements RNN from scratch because of off-the-shelf RNN restrictions to use
    the output size identical to hidden size, or appending an additional layer,
    thereby changing the behaviour.
    """

    def __init__(self, input_size, hidden_size, n_targets=1):
        super(BaselineRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = self.input_size
        self.n_targets = n_targets + (
            1 if n_targets > 1 else 0)  # special eos symbol

        # note: i2h can be replicated by off-the-shelf RNN without the loop in
        # training, but with the loop for output generation.
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, self.output_size)

    def forward(self, sequences):
        """
        Args:
            sequences: (batch, seq_len, input_size) the DataLoader batch.
        Returns:
            output: (batch, n_outputs, output_size) where output_size =
            max_token_id
            hidden: (batch, num_layers=1, hidden_size)
        """

        # [batch_size, hidden_size]
        hidden = torch.zeros(sequences.size(0), self.hidden_size)
        # [batch_size, n_targets, output_size=max_token_id]
        output = torch.zeros(sequences.size(0), self.n_targets,
                             self.output_size)

        # off-the-shelf RNN would remove this loop;
        # the outputs would then be computed on the `outputs` variable of
        # that RNN
        for i in range(
                sequences.size(1) - self.n_targets):  # iterate over timesteps
            combined = torch.cat((sequences[:, i, :], hidden), dim=1)
            hidden = self.i2h(combined)
            hidden = F.tanh(hidden)

        for i in range(self.n_targets):
            combined = torch.cat((sequences[:, -self.n_targets + i, :],
                                  hidden), dim=1)
            hidden = self.i2h(combined)
            hidden = F.tanh(hidden)
            output[:, -self.n_targets + i] = self.h2o(hidden)

        return output, hidden

    def reset_parameters(self):
        self.i2h.reset_parameters()
        self.h2o.reset_parameters()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
