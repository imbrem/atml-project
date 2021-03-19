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

import torch.nn as nn


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
                                hidden_size=hidden_size,
                                batch_first=True)

    def forward(self, input, n_outputs=1):
        """
        Args:
            input: (batch, seq_len, input_size)
        Returns:
            output: (batch, n_outputs, hidden_size)
            hidden: (batch, num_layers, hidden_size)
        """
        # initial hidden representation defaults to 0
        output, hidden = self.rnn_layer(input)
        return output[:, -n_outputs:, :], hidden

    def reset_parameters(self):
        self.rnn_layer.reset_parameters()
