""" Prepares baseline RNN/LSTM data. 

Creates a PyTorch DataLoader with batched sequences.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data


def is_uniform_length(dataset):
    uniform_length = True
    seq_len = len(dataset[0])
    for i in range(1, len(dataset)):
        if seq_len != len(dataset[i]):
            uniform_length = False
            break
    return uniform_length


def get_max_sequence_length(dataset):
    return max(map(len, dataset))


def load_rnn_data_from_file(file_name, n_targets=1):
    dataset = []
    with open(file_name, 'r') as f:
        for line in f:
            example = map(int, line.split())
            dataset.append(example)

        if is_uniform_length(dataset):
            data = torch.tensor(dataset)
            seq = data[:, :data.size(1)-n_targets]
            target = data[:, data.size(1)-n_targets:]

            if n_targets == 1:
                return seq, target
            else:  # extend sequence, append special end target
                ext_seq = torch.Tensor(seq.size(0), seq.size(1)+n_targets)
                ext_seq[:, :-n_targets] = seq
                ext_seq[:, -n_targets:] = ext_seq.narrow(
                    1, seq.size(1)-1, 1).repeat(1, n_targets)

                ext_target = torch.Tensor(seq.size(0), n_targets+1)
                ext_target[:, :-1] = target
                # append special end symbol
                ext_target[:, n_targets] = target.max() + 1
                return ext_seq, ext_target

        else:  # sequence length not equal
            target = []
            seq = []
            max_target = 0

            for i in range(len(dataset)):
                s = torch.tensor(dataset[i])
                s.resize_(1, s.nelement())

                if n_targets == 1:
                    seq.append(s[:, :s.size(1)-n_targets])
                    target.append(s[:, s.size(1)-n_targets:])
                else:
                    # TODO why does it only extend it by 1?
                    # TODO why is sequence extended
                    seq.append(torch.Tensor(1, s.size(1)+1))
                    seq[i][:, :s.size(1)-n_targets] = s[:,
                                                        :s.size(1)-n_targets]
                    seq[i][:, s.size(1)-n_targets:] = s[:, s.size(1)-n_targets]

                    # last entry will be the special end symbol
                    target.append(torch.Tensor(1, n_targets+1))
                    target[i][:, :n_targets] = s[:,
                                                 s.size(1)-n_targets:s.size(1)]

                    t_max = target[i][0, :n_targets].max()
                    if t_max > max_target:
                        max_target = t_max

            # append special end symbol
            if n_targets != 1:
                for i in range(len(dataset)):
                    target[i][:, -1] = max_target + 1

            # TODO shouldn't these be tensors
            return seq, target


def find_max_in_list_of_tensors(lst):
    return max(list(map(torch.max, lst)))


# TODO translate
def split_set_tensor(x_train, t_train, n_train, n_val, some_boolean):
    pass

# TODO translate


def split_set_input_output(x_train, t_train, n_train, n_val, some_boolean):
    pass
