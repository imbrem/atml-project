"""
Data preprocessing module.

Converts the bAbI dataset files into tensors ready for training.

This is not original code, but is adapted from the bAbI dataset processing code from
https://github.com/chingyaoc/ggnn.pytorch/blob/master/utils/data/dataset.py and
https://github.com/chingyaoc/ggnn.pytorch/blob/master/utils/data/dataloader.py

RNN processing part adapts the original code at
https://github.com/yujiali/ggnn/blob/master/babi/babi_data.lua
"""


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

# GRAPH DATA


def load_graphs_from_file(file_name):
    data_list = []
    edge_list = []
    target_list = []
    with open(file_name, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                data_list.append([edge_list, target_list])
                edge_list = []
                target_list = []
            else:
                digits = []
                line_tokens = line.split(" ")
                if line_tokens[0] == "?":
                    for i in range(1, len(line_tokens)):
                        digits.append(int(line_tokens[i]))
                    target_list.append(digits)
                else:
                    for i in range(len(line_tokens)):
                        digits.append(int(line_tokens[i]))
                    edge_list.append(digits)
    return data_list


def find_max_edge_id(data_list):
    max_edge_id = 0
    for data in data_list:
        edges = data[0]
        for item in edges:
            if item[1] > max_edge_id:
                max_edge_id = item[1]
    return max_edge_id


def find_max_node_id(data_list):
    max_node_id = 0
    for data in data_list:
        edges = data[0]
        for item in edges:
            if item[0] > max_node_id:
                max_node_id = item[0]
            if item[2] > max_node_id:
                max_node_id = item[2]
    return max_node_id


def find_max_task_id(data_list):
    max_node_id = 0
    for data in data_list:
        targe = data[1]
        for item in targe:
            if item[0] > max_node_id:
                max_node_id = item[0]
    return max_node_id


def split_set(data_list):
    n_examples = len(data_list)
    idx = range(n_examples)
    train = idx[:50]
    val = idx[-50:]
    return np.array(data_list, dtype=object)[train], np.array(data_list, dtype=object)[val]


def data_convert(data_list, n_annotation_dim):
    n_nodes = find_max_node_id(data_list)
    n_tasks = find_max_task_id(data_list)
    task_data_list = []
    for i in range(n_tasks):
        task_data_list.append([])
    for item in data_list:
        edge_list = item[0]
        target_list = item[1]
        for target in target_list:
            task_type = target[0]
            task_output = target[-1]
            annotation = np.zeros([n_nodes, n_annotation_dim])
            annotation[target[1]-1][0] = 1
            task_data_list[task_type -
                           1].append([edge_list, annotation, task_output])
    return task_data_list


def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] = 1
        a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1
    return a


def create_pg_graph(datapoint, n_edge_types):
    """Convert a graph to pytorch geometric form"""
    edges, annotations, target = datapoint
    x = torch.FloatTensor(annotations)
    directed_edge_index = torch.LongTensor(
        [[edge[0]-1, edge[2]-1] for edge in edges])
    reverse_edge_index = torch.index_select(
        directed_edge_index, 1, torch.LongTensor([1, 0]))
    edge_index = torch.cat([directed_edge_index, reverse_edge_index], dim=0).T
    # print("Edge index", edge_index)

    edge_type_indices = torch.LongTensor(
        [[i, edge[1]-1] for i, edge in enumerate(edges)])
    # print("Edge type", edge_type_indices)
    reverse_edge_type_indices = torch.LongTensor(
        [[i+len(edges), edge[1]-1+n_edge_types] for i, edge in enumerate(edges)])
    full_edge_type_indices = torch.cat(
        [edge_type_indices, reverse_edge_type_indices], dim=0)
    # print("Full edge", full_edge_type_indices)

    edge_attr = torch.zeros(edge_index.size(1), n_edge_types * 2)
    for edge_type in full_edge_type_indices.numpy().tolist():
        edge_attr[tuple(edge_type)] = 1

    y = torch.FloatTensor([datapoint[2]-1]).view(1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class bAbIDataset:
    """
    Load bAbI tasks for GGNN
    """

    def __init__(self, path, task_id, is_train):
        all_data = load_graphs_from_file(path)
        self.n_edge_types = find_max_edge_id(all_data)
        self.n_tasks = find_max_task_id(all_data)
        self.n_node = find_max_node_id(all_data)

        all_task_train_data, all_task_val_data = split_set(all_data)

        if is_train:
            all_task_train_data = data_convert(all_task_train_data, 1)
            self.data = all_task_train_data[task_id]
        else:
            all_task_val_data = data_convert(all_task_val_data, 1)
            self.data = all_task_val_data[task_id]

    def __getitem__(self, index):
        return create_pg_graph(self.data[index], n_edge_types=self.n_edge_types)

    def __len__(self):
        return len(self.data)

# RNN DATA

# TODO check all off-by-1s (ugh Lua's 1-indexing....)


def load_rnn_data_from_file(file_name, n_targets=1):
    dataset = []
    with open(file_name, 'r') as f:
        for line in f:
            example = map(int, line.split())
            dataset.append(example)

        uniform_length = True
        seq_len = len(dataset[0])
        for i in range(1, len(dataset)):
            if seq_len != len(dataset[i]):
                uniform_length = False
                break

        if uniform_length:
            data = torch.Tensor(dataset)
            seq = data.narrow(1, 0, data.size(1)-n_targets)
            target = data.narrow(1, data.size(1)-n_targets, n_targets)

            if n_targets == 1:
                return seq, target
            else:  # extend sequence, append special end target
                ext_seq = torch.Tensor(seq.size(0), seq.size(1)+n_targets)
                # TODO not sure about assignment after copy
                ext_seq = ext_seq.narrow(1, 0, seq.size(1)).copy(seq)
                # TODO assignment?
                torch.repeatTensor(ext_seq.narrow(1, seq.size(1), n_targets), seq.narrow(
                    1, seq.size(1), 0), 0, n_targets)

                ext_target = torch.Tensor(seq.size(0), n_targets)
                # TODO not sure about assignment after copy
                ext_target = ext_target.narrow(1, 0, n_targets).copy(target)
                # TODO required assignment? `narrow` is not in place
                # append special end symbol
                ext_target.narrow(1, n_targets, 0).fill(
                    target.max()+1)  # TODO +1?
                return ext_seq, ext_target
        else:  # sequence length not equal
            target = []
            seq = []
            max_target = 0

            for i in range(len(dataset)):
                s = torch.Tensor(dataset[i])
                s = s.resize(0, s.nelement())

                if n_targets == 1:
                    seq[i] = s.narrow(1, 0, s.size(1)-n_targets)
                    target[i] = s.narrow(1, s.size(1)-n_targets, n_targets)
                else:
                    seq[i] = torch.Tensor(1, s.size(1)+1)  # TODO +1??
                    seq[i].narrow(1, 0, s.size(
                        1)-n_targets).copy(s.narrow(1, 0, s.size(1)-n_targets))
                    seq[i].narrow(
                        1, s.size(1)-n_targets, n_targets).fill(s[s.size(1)-n_targets+1])  # TODO +1?

                    target[i] = torch.Tensor(1, n_targets+1)
                    target[i].narrow(1, 0, n_targets).copy(
                        s.narrow(1, s.size(1)-n_targets, n_targets))

                    t_max = target[i].narrow(1, 0, n_targets).max()
                    if t_max > max_target:
                        max_target = t_max

            # append special end symbol
            if n_targets != 1:
                for i in range(len(dataset)):
                    target[i][target[i].nelement()] = max_target + 1

            return seq, target


# TODO replace with a standard tensor library function
def find_max_in_list_of_tensors(list):
    max = list[0][0]
    for v in list:
        m = v.max()
        if m > max:
            max = m
    return max


if __name__ == "__main__":
    from torch_geometric.data import DataLoader

    dataroot = 'babi_data/processed_1/train/4_graphs.txt'
    train_dataset = bAbIDataset(dataroot, 0, True)
    loader = DataLoader(train_dataset, batch_size=2)
    batch0 = next(iter(loader))
    print(batch0.x, batch0.edge_index, batch0.batch, batch0.edge_attr, batch0.y)
