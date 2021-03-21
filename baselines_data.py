""" Prepares baseline RNN/LSTM data. 

Creates a PyTorch DataLoader with batched sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


def load_rnn_data_from_file(file_name, n_targets=1):
    """ Returns sequences and targets in a given file as tensors. """
    sequence_list = []
    target_list = []
    with open(file_name, 'r') as f:
        for line in f:
            example = map(int, line.split())
            sequence_list.append(torch.tensor(example[:-n_targets]))
            target_list.append(example[-n_targets:])

    sequences = torch.nn.utils.rnn.pad_sequence(sequence_list)
    targets = torch.tensor(target_list)

    # Append special end of sequence symbol when the target is a sequence.
    if n_targets != 1:
        eos_id = (targets.max() + 1) * torch.ones(targets.size(0), 1)
        targets = torch.cat((targets, eos_id), dim=1)

    return sequences, targets


class bAbIRNNDataset(Dataset):
    """ Load bAbI dataset for RNN. """

    # TODO arguments based on path, task_id, whether training dataset
    def __init__(self, data_file, n_targets=1):
        """
        Args:
            data_file (string): Path to the file with the task data.
            root_dir (string): Dataset directory.
        """
        self.sequences, self.targets = load_rnn_data_from_file(
            data_file, n_targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx, :], "target": self.targets[idx, :]}


def split_set_tensor(x_train, t_train, n_train, n_val, some_boolean):
    pass

# TODO translate


def split_set_input_output(x_train, t_train, n_train, n_val, some_boolean):
    pass
