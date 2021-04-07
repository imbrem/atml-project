# A GGSNN performing per-node output
# Can be used for node-selection purposes by using a per-node output of size 1
import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
from typing import Union, List, Optional
from ggnns.per_node_ggnn import PerNodeLayer, make_per_node_hidden_layer
from ggnns.base_ggnn import make_ggnn
from torch.nn import Module


class PerNodeGGSNN(Module):
    def __init__(
        self,
        annotation_size: int,
        output_size: int,
        output_num_layers: int,
        propagation_num_layers: Option[int] = None,
        propagation_hidden_state: int = 0,
        output_hidden_state: int = 0,
        output_hidden_layer: Option[Module] = None,
        propagation_hidden_layer: Option[Module] = None,
        linear_activation: Module = nn.ReLU(inplace=True),
        padding_mode: str = 'constant',
        padding_const: int = 0,
        graph_attention_impl: str = 'torch_geometric',
        ggnn_impl: str = 'torch_geometric',
        separate_propagation: bool = False,
        **kwargs,
    ):
        super(GraphLevelGGSNN, self).__init__()
        self.annotation_size = annotation_size
        self.output_size = output_size
        self.output_num_layers = output_num_layers
        if propagation_num_layers is None:
            self.propagation_num_layers = output_num_layers
        else:
            assert separate_propagation or propagation_num_layers == output_num_layers
            self.propagation_num_layers = propagation_num_layers
        assert separate_propagation or propagation_hidden_state == output_hidden_state
        self.propagation_hidden_state = propagation_hidden_state
        self.output_hidden_state = output_hidden_state
        self.padding_mode = padding_mode
        self.padding_const = padding_const
        self.separate_propagation = separate_propagation

        if separate_propagation:
            self.output_ggnn_layer = make_ggnn(
                state_size=annotation_size + self.output_hidden_state,
                num_layers=output_num_layers,
                ggnn_impl=ggnn_impl, **kwargs)
        self.propagation_ggnn_layer = make_ggnn(
            state_size=annotation_size + self.propagation_hidden_state,
            num_layers=propagation_num_layers,
            ggnn_impl=ggnn_impl, **kwargs)
        self.output_hidden_layer = make_per_node_hidden_layer(
            hidden_layer=output_hidden_layer,
            annotation_size=annotation_size,
            hidden_state=output_hidden_state,
            output_size=output_size
        )
        self.propagation_hidden_layer = propagation_hidden_layer(
            hidden_layer=propagation_hidden_layer,
            annotation_size=annotation_size,
            hidden_state=propagation_hidden_state,
            output_size=annotation_size
        )

    def forward(self, x, edge_index, batch, pred_steps, **kwargs):
        outputs = []
        for k in range(pred_steps):
            # Step 1: pad `x` from `annotation_size` to `hidden_state + annotation_size`
            assert x.shape[-1] == self.annotation_size
            x_ggnn = nn.functional.pad(
                x, (0, self.hidden_state), self.padding_mode, self.padding_const)
            assert x_ggnn.shape[-1] == self.annotation_size + \
                self.hidden_state
            # Step 2: evolve the graph for the k-th pred_step
            x_ggnn = self.propagation_ggnn_layer(x_ggnn, edge_index, **kwargs)
            # Step 3: catenate the original input to the propagated output
            x_ggnn = torch.cat((x_ggnn, x), -1)
            # Step 4: compute output, using cached propagation output if separate propagation is disabled
            if self.separate_propagation:
                out = self.output_ggnn_layer(x, edge_index, **kwargs)
                out = torch.cat((out, x), -1)
            else:
                out = x_ggnn
            out = self.output_hidden_layer(out)
            outputs.append(out)
            del out
            # Step 5: transform the propagated output, now of shape `hidden_state + 2*annotation_size`, to a new set of annotations
            x = self.propagation_hidden_layer(x_ggnn)
            del x_ggnn

        return torch.stack(outputs, dim=1)
