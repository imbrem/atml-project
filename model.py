import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F


class GlobalGGNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers=32, aggr='add', bias=True, **kwargs):
        super(GlobalGGNN, self).__init__()
        self.gated_graph_layer = torch_geometric.nn.GatedGraphConv(
            out_channels=input_size, num_layers=num_layers, aggr=aggr, bias=bias, **kwargs)
        self.final_layer = torch.nn.Linear(input_size, output_size)

    def forward(self, x, edge_index, batch):
        x = self.gated_graph_layer(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.final_layer(x)
        x = torch.sigmoid(x)

        return x

    def reset_parameters(self):
        self.gated_graph_layer.reset_parameters()
        self.final_layer.reset_parameters()


class GGSNN(nn.Module):
    pass
