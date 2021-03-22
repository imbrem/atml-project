""" Prepares baseline RNN/LSTM data. 

Creates a PyTorch DataLoader with batched sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


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


class BabiRNNDataset(Dataset):
    """ Load bAbI dataset for RNN. """

    def __init__(self, root_dir, task_id, n_targets=1, validation=False, test=False):
        """
        Args:
            root_dir (string): Path to the fold root directory.
            task_id (int): The ID of the bAbI task.
            n_targets (int): The length of output sequences.
            validation: Whether to load the validation data.
            test: Whether to load the test data.
        """
        data_file = Path(root_dir)

        data_file = data_file / ('test' if test else 'train')
        data_file = data_file / '{}_rnn.txt'.format(task_id)

        if validation and not test:
            data_file = data_file / '.val'

        self.sequences, self.targets = load_rnn_data_from_file(data_file, n_targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {'x': self.sequences[idx, :], 'y': self.targets[idx, :]}

# TODO one-hot encoding

# TODO training and validation sets
# split training data into train & val
# checking if validation data file exists
# if Path(args.data_file + '.val').is_file():
#     print('Validation file exists\nSplitting part of training data for validation.')
#     # TODO split_set_tensor
#     seq_train, target_train, seq_val, target_val = split_set_tensor(
#         seq_train, target_train, args.n_train, args.n_val, True)
# else:
#     # TODO isn't this behaving in the opposite way
#     # if n_train is 0, automatically use all the training data available
#     if args.n_train:
#         seq_train, target_train = split_set_tensor(
#             seq_train, target_train, args.n_train, 0, True)

#     print('Loading validation data from {}.val'.format(args.data_file))
#     seq_val, target_val = load_rnn_data_from_file(
#         args.data_file + '.val', args.n_targets)
