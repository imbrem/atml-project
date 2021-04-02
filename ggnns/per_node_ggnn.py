# An implementation of a GGNN layer which returns one output for each node
import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
from typing import Union, List, Optional
from base_ggnn import make_ggnn
from torch.nn import Module


class PerNodeGGNN(Module):
    def __init__(self,
                 annotation_size: int,
                 output_size: int,
                 num_layers: int,
                 hidden_state: int = 0,
                 hidden_layer: Optional[Module] = None,
                 batched_hidden_layer: bool = False,
                 padding_mode: str = 'constant',
                 padding_const: int = 0,
                 ggnn_impl: str = 'torch_geometric', **kwargs):
        super(PerNodeGGNN, self).__init__()

        self.hidden_state = hidden_state
        self.annotation_size = annotation_size
        self.output_size = output_size

        self.ggnn_layer = make_ggnn(
            state_size=annotation_size + hidden_state,
            num_layers=num_layers,
            ggnn_impl=ggnn_impl,
            **kwargs,
        )

        self.padding_mode = padding_mode
        self.padding_const = padding_const
        self.batched_hidden_layer = batched_hidden_layer

        if hidden_layer is None:
            self.hidden_layer = nn.Sequential(
                nn.Linear(annotation_size + hidden_state +
                          annotation_size, output_size),
            )
        else:
            self.hidden_layer = hidden_layer

    def forward(self, x, edge_index, batch, **kwargs):
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
        if self.batched_hidden_layer:
            x = self.hidden_layer(x, batch=batch, **kwargs)
        else:
            x = self.hidden_layer(x, **kwargs)
        assert x.shape[-1] == self.output_size
        return x

    def reset_parameters(self):
        self.ggnn_layer.reset_parameters()
        self.hidden_layer.reset_parameters()


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
