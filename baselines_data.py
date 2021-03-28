"""
Prepares baseline RNN/LSTM data.

Creates a PyTorch DataLoader with batched sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def get_data_filename(root_dir, fold_id, task_id, split):
    """ Returns the filename up to .txt extension. """
    filename = Path(root_dir)
    filename = filename / 'processed_{}'.format(fold_id)
    filename = filename / 'rnn'
    filename = filename / ('test' if split is 'test' else 'train')
    filename = filename / '{}_rnn.txt'.format(task_id)

    return filename


def get_sequence_and_target_lists_from_file(filename, n_targets=1, n_train=0):
    """ Returns sequences and targets in a given file as lists of token IDs. """
    sequence_list = []
    target_list = []
    with open(filename, 'r') as f:
        n_lines = 0
        for line in f:
            example = list(map(int, line.split()))
            sequence_list.append(torch.tensor(example[:-n_targets]))
            target_list.append(torch.tensor(example[-n_targets:]))
            n_lines += 1
            if 0 < n_train == n_lines:
                break

    return sequence_list, target_list


def get_max_token_id(filename, n_targets):
    filename = filename.parent / (filename.name + '.dict')
    n_lines = 0
    with open(filename, 'r') as f:
        for _ in f:
            n_lines += 1

    return n_lines + (1 if n_targets > 1 else 0)


def one_hot_token_list(token_list, max_token_id):
    one_hot_list = []
    for seq in token_list:
        y = (seq.clone().detach() - 1).view(-1, 1)
        y_onehot = torch.FloatTensor(y.size(0), max_token_id).zero_()
        y_onehot.scatter_(1, y, 1)
        one_hot_list.append(y_onehot)

    return one_hot_list


def transform_sequence_list(sequence_list, max_token_id):
    one_hot_sequence_list = one_hot_token_list(sequence_list, max_token_id)
    sequences = torch.nn.utils.rnn.pad_sequence(one_hot_sequence_list,
                                                batch_first=True)
    return sequences


def transform_target_list(target_list, n_targets, max_token_id):
    # Append special end of sequence symbol when the target is a sequence.
    one_hot_target_list = one_hot_token_list(target_list, max_token_id)
    targets = torch.stack(
        one_hot_target_list)  # [n_seq x n_targets x max_token_id]
    if n_targets > 1:
        eos_tensor = torch.zeros(targets.size(0), 1, targets.size(2))
        eos_tensor[:, :, -1] = 1
        targets = torch.cat((targets, eos_tensor),
                            dim=1)  # [n_seq x (n_targets + 1) x max_token_id]

    return targets


def prepare_sequences_and_targets(token_lists, n_targets, max_token_id):
    """ Transforms token ID lists into tensors.

    Adds zero-vector padding to the end of sequence if the sequences have
    different lengths.
    Transforms tokens into one-hot encoded vectors.
    Appends an end-of-sequence token to the target if n_targets > 1.
    """
    sequence_list, target_list = token_lists
    sequences = transform_sequence_list(sequence_list, max_token_id)
    targets = transform_target_list(target_list, n_targets, max_token_id)

    return sequences, targets


# TODO use custom number of training/validation examples
class BabiRNNDataset(Dataset):
    """ Loads bAbI dataset for RNN.

    Obtains the file (training, validation, or test) for the corresponding
    fold from the root directory containing the entire dataset (will likely
    be babi_data/ in this case).

    Adds zero-padding at the end of the sequence if sequences in the dataset
    have different lengths. Adds a special end-of-sequence token for the
    targets that consist of multiple outputs. All tokens are one-hot encoded.
    """

    def __init__(self, root_dir, fold_id, task_id, n_targets=1,
                 split='train', n_train=0):
        """
        Args:
            root_dir (string): Path to the fold root directory.
            fold_id (int): The number of the fold.
            task_id (int): The ID of the bAbI task.
            n_targets (int): The length of output sequences.
            split (str): One of 'train', 'validation', 'test'
            n_train (int): How many training instances to use.
        """

        filename = get_data_filename(root_dir, fold_id, task_id, split)
        self.max_token_id = get_max_token_id(filename, n_targets)

        if split is 'validation':
            filename = filename.parent / (filename.name + '.val')

        if split is not 'train':
            n_train = 0

        data = get_sequence_and_target_lists_from_file(filename, n_targets,
                                                       n_train)
        self.sequences, self.targets = \
            prepare_sequences_and_targets(data, n_targets, self.max_token_id)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx, :], self.targets[idx, :]
