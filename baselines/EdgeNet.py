import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor

from ggnn_data import bAbIDataset


class EdgeNet(MessagePassing):
    def __init__(self, annotation_size, edge_attr_size):
        super(EdgeNet, self).__init__()
        self.annotation_size = annotation_size
        self.edge_to_annotation = nn.Linear(edge_attr_size, annotation_size)
        self.activation = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(annotation_size, annotation_size),
            nn.ReLU(inplace=True),
            nn.Linear(annotation_size, annotation_size),
            nn.ReLU(inplace=True)
        )
        self.gate_nn = nn.Linear(annotation_size, 1)

    def padding(self, x_j):
        if x_j.size(1) == self.annotation_size:
            return x_j
        return torch.cat([torch.zeros((x_j.size(0), self.annotation_size-x_j.size(1))), x_j], dim=1)

    def message(self, x_j, edge_attr):
        x_j = self.padding(x_j)
        msg = x_j if edge_attr is None else x_j + self.activation(self.edge_to_annotation(edge_attr))
        return msg

    def forward(self, x, edge_index, edge_attr, batch):
        size = batch[-1].item() + 1
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.gate_nn(self.mlp(out))
        return softmax(out, batch, num_nodes=size).view(size, -1)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    edge_net = EdgeNet(annotation_size=4, edge_attr_size=4)
    optimizer = optim.Adam(edge_net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    from torch_geometric.data import DataLoader

    dataroot = 'babi_data/processed_1/train/4_graphs.txt'
    train_dataset = bAbIDataset(dataroot, 0, True)
    loader = DataLoader(train_dataset, batch_size=2)
    for _ in range(10):
        epoch_loss = 0
        epoch_correct = 0
        total = 0
        for batch in loader:
            prediction_probs = edge_net(x=batch.x, edge_index=batch.edge_index,
                                        edge_attr=batch.edge_attr, batch=batch.batch)
            _, predicted = torch.max(prediction_probs, 1)

            loss = criterion(prediction_probs, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            examples = batch.y.size(0)
            total += examples
            epoch_loss += loss.item() * examples
            epoch_correct += (predicted == batch.y).sum()
        print("Epoch loss: {}; Epoch accuracy: {}".format(epoch_loss/total, epoch_correct/total))
