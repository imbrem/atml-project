"""
Baseline RNN.

Contains the definition of the BaselineRNN class.

Trained on symbolic data in raw sequence form using cross-entropy loss.

The implementation in the paper uses 50-dimensional embeddings and
50-dimensional hidden layers, predicting a single output, which is
treated as in a classification problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineRNN(nn.Module):
    """Baseline RNN class.

    Implemented using a single off-the-shelf PyTorch RNN layer, which
    could be modified to include the complete low-level detail if needed.
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super(BaselineRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_layer = nn.RNN(input_size=input_size,
                                hidden_size=hidden_size)

    def forward(self, input):
        # TODO what processing do I need here
        output, hidden = self.rnn_layer(input)
        # TODO scoring / classification
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)

    def reset_parameters(self):
        self.rnn_layer.reset_parameters()
