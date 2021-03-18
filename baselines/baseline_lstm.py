"""
Baseline LSTM.

Contains the definition of the BaselineLSTM class.

Trained on symbolic data in raw sequence form using cross-entropy loss.

The implementation in the paper uses 50-dimensional embeddings and
50-dimensional hidden layers, predicting a single output, which is
treated as in a classification problem.
"""

import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """Baseline LSTM class.

    Implemented using a single off-the-shelf PyTorch LSTM layer, which
    could be modified to include the complete low-level detail if needed.
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super(BaselineLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layer = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size)

    def forward(self, input):
        # TODO processing the sequence into embeddings
        output, hidden = self.lstm_layer(input)
        # TODO derive scoring of outputs?
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)

    def reset_parameters(self):
        self.lstm_layer.reset_parameters()
