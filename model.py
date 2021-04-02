import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F


def gated_graph_conv_layer(input_size, output_size, num_layers, aggr='add',
                           impl='torch', bias=True, **kwargs):
    """
    Construct a gated graph convolutional layer using a given implementation
    """
    if impl == 'torch':
        return torch_geometric.nn.GatedGraphConv(
            out_channels=input_size, num_layers=num_layers, aggr=aggr,
            bias=bias, **kwargs)
    else:
        raise ValueError(f"Invalid implementation \"{impl}\"")


class GlobalGGNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers=32, aggr='add',
                 bias=True, **kwargs):
        super(GlobalGGNN, self).__init__()
        self.gated_graph_layer = torch_geometric.nn.GatedGraphConv(
            out_channels=input_size, num_layers=num_layers, aggr=aggr,
            bias=bias, **kwargs)
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
    def __init__(self, x_gated_graph, o_gated_graph):
        super(GGSNN, self).__init__()
        self.x_gated_graph = x_gated_graph
        self.o_gated_graph = o_gated_graph
