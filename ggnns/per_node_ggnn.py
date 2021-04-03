# An implementation of a GGNN layer which returns one output for each node
import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
from typing import Union, List
from ggnns.base_ggnn import make_ggnn
from torch.nn import Module


class PerNodeGGNN(Module):
    def __init__(self,
                 annotation_size: int,
                 output_size: int,
                 num_layers: int,
                 hidden_state: int = 0,
                 hidden_layers: Union[List[int], int] = [],
                 linear_activation: Module = nn.ReLU(inplace=True),
                 padding_mode: str = 'constant',
                 padding_const: int = 0,
                 ggnn_impl: str = 'torch_geometric', **kwargs):
        super(PerNodeGGNN, self).__init__()

        self.hidden_state = hidden_state
        self.annotation_size = annotation_size

        self.ggnn_layer = make_ggnn(
            state_size=annotation_size + hidden_state,
            num_layers=num_layers,
            ggnn_impl=ggnn_impl,
            **kwargs,
        )

        self.padding_mode = padding_mode
        self.padding_const = padding_const
        self.per_node_layer = PerNodeLayer(
            input_size=annotation_size + hidden_state + annotation_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            linear_activation=linear_activation
        )

    def forward(self, x, edge_index, **kwargs):
        # Step 1: pad `x` from `annotation_size` to `hidden_state +
        # annotation_size`
        assert x.shape[-1] == self.annotation_size
        x_ggnn = nn.functional.pad(
            x, (0, self.hidden_state), self.padding_mode, self.padding_const)
        assert x_ggnn.shape[-1] == self.annotation_size + \
               self.hidden_state
        # Step 2: pass the padded `x` into the GGNN layer
        x_ggnn = self.ggnn_layer(x, edge_index, **kwargs)
        # Step 3: catenate the GGNN output with the original input
        x = torch.cat((x_ggnn, x), -1)
        del x_ggnn
        assert x.shape[-1] == self.annotation_size + \
               self.hidden_state + self.annotation_size
        # Step 4: pass this through the per-node linear adapter
        x = self.per_node_layer(x)

        return x

    def reset_parameters(self):
        self.ggnn_layer.reset_parameters()
        self.per_node_layer.reset_parameters()


class PerNodeLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_layers: Union[List[int], int] = [],
                 linear_activation=nn.ReLU(inplace=True)):
        super(PerNodeLayer, self).__init__()

        self.linear_activation = linear_activation
        self.input_size = input_size
        self.output_size = output_size

        linear_layers = []
        for size in hidden_layers:
            linear_layers.append(nn.Linear(input_size, size))
            input_size = size
        linear_layers.append(nn.Linear(input_size, output_size))

        self.linear_layers = nn.ModuleList(linear_layers)

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
            x = self.linear_activation(x)

        return x

    def reset_parameters(self):
        for layer in self.linear_layers:
            layer.reset_parameters()


if __name__ == "__main__":
    print("Identity test for PerNodeGGNN")
    NO_ATTRIBUTES = 5
    NO_CYCLES = 10
    print(
        f"Generating cycle data (attributes = {NO_ATTRIBUTES}, cycles = "
        f"{NO_CYCLES})...")
    from cycle_data import *

    cycles = [
        id_graph(g, NO_ATTRIBUTES) for g in generate_cycles(NO_CYCLES)
    ]
    print(f"Generated {len(cycles)} cycles")
    data_loader = torch_geometric.data.DataLoader(cycles, batch_size=1)
    print("Constructing GGNN...")
    ggnn = PerNodeGGNN(NO_ATTRIBUTES, NO_ATTRIBUTES, 2, hidden_state=10).cuda()
    print(f"GGNN: {ggnn}")
    print("Setting up training...")
    opt = torch.optim.Adam(ggnn.parameters(), lr=0.01)
    losses = train(ggnn, data_loader, opt, torch.nn.MSELoss())
    print("Plotting losses...")
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.show()
    print("Examples:")
    import random

    choices = random.choices(cycles, k=3)
    for choice in choices:
        print(
            f"x = {choice.x} ==> ggnn(x) = "
            f"{ggnn(choice.x.cuda(), choice.edge_index.cuda())}")
