"""
Data preprocessing module.

Converts the bAbI dataset files into tensors ready for training.

The code is adapted from the bAbI dataset processing code at:
https://github.com/chingyaoc/ggnn.pytorch/blob/master/utils/data/dataset.py
https://github.com/chingyaoc/ggnn.pytorch/blob/master/utils/data/dataloader.py
"""

import numpy as np
import torch
from torch_geometric.data import Data
import baselines_data


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
    return np.array(data_list, dtype=object)[train], \
           np.array(data_list, dtype=object)[val]


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
            annotation[target[1] - 1][0] = 1
            task_data_list[task_type -
                           1].append([edge_list, annotation, task_output])
    return task_data_list


def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx - 1][(e_type - 1) * n_nodes + src_idx - 1] = 1
        a[src_idx - 1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1
    return a


def create_pg_graph(datapoint, n_edge_types):
    """Convert a graph to pytorch geometric form"""
    edges, annotations, target = datapoint
    x = torch.FloatTensor(annotations)
    directed_edge_index = torch.LongTensor(
        [[edge[0] - 1, edge[2] - 1] for edge in edges])
    reverse_edge_index = torch.index_select(
        directed_edge_index, 1, torch.LongTensor([1, 0]))
    edge_index = torch.cat([directed_edge_index, reverse_edge_index], dim=0).T
    # print("Edge index", edge_index)

    edge_type_indices = torch.LongTensor(
        [[i, edge[1] - 1] for i, edge in enumerate(edges)])
    # print("Edge type", edge_type_indices)
    reverse_edge_type_indices = torch.LongTensor(
        [[i + len(edges), edge[1] - 1 + n_edge_types] for i, edge in
         enumerate(edges)])
    full_edge_type_indices = torch.cat(
        [edge_type_indices, reverse_edge_type_indices], dim=0)
    # print("Full edge", full_edge_type_indices)

    edge_attr = torch.zeros(edge_index.size(1), n_edge_types * 2)
    for edge_type in full_edge_type_indices.numpy().tolist():
        edge_attr[tuple(edge_type)] = 1

    y = torch.FloatTensor([datapoint[2] - 1]).view(1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def get_sequential_graph(sequence, target):
    # sequence: [seq_len, max_node_id]
    # target: [n_targets]
    edge_index = torch.tensor(
        (range(sequence.size(0)), range(1, sequence.size(0) + 1)),
        dtype=torch.long)
    return Data(x=sequence, edge_index=edge_index, y=target)


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


class BabiSequentialGraphDataset:
    """ Loads the sequential RNN data as a linear graph for GGS-NN training.

    Obtains the file (training, validation, or test) for the corresponding
    fold from the root directory containing the entire dataset (will likely
    be babi_data/ in this case).
    """

    def __init__(self, root_dir, fold_id, task_id, n_targets=1,
                 split='train', n_train=0):
        """
        Args:
            root_dir (str): Path to the dataset root directory ('babi_data/')
            fold_id (int): The number of the fold.
            task_id (int): The ID of the bAbI task.
            n_targets (int): The length of output sequences.
            split (str): One of 'train', 'validation', 'test'
            n_train (int): How many training instances to use.
        """

        filename = baselines_data.get_data_filename(root_dir, fold_id,
                                                    task_id, split)
        max_token_id = baselines_data.get_max_token_id(root_dir, task_id,
                                                       n_targets)

        if split is 'validation':
            filename = filename.parent / (filename.name + '.val')

        if split is not 'train':
            n_train = 0

        sequences, targets = \
            baselines_data.get_sequence_and_target_lists_from_file(
                filename,
                n_targets,
                n_train)

        one_hot_sequences = baselines_data.one_hot_token_list(sequences,
                                                              max_token_id)

        self.graphs = []
        for sequence, target in zip(one_hot_sequences, targets):
            self.graphs.append(get_sequential_graph(sequence, target))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.graphs[idx]


if __name__ == "__main__":
    from torch_geometric.data import DataLoader

    dataroot = 'babi_data/processed_1/train/4_graphs.txt'
    train_dataset = bAbIDataset(dataroot, 0, True)
    loader = DataLoader(train_dataset, batch_size=2)
    batch0 = next(iter(loader))
    print(batch0.x, batch0.edge_index, batch0.batch, batch0.edge_attr, batch0.y)
