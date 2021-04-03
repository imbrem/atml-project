# An implementation of a GGNN layer which produces graph-level output

import torch
import torch.nn.functional
from typing import Optional
from ggnns.global_soft_attention import make_graph_attention
from ggnns.base_ggnn import make_ggnn
from torch.nn import Module


class GraphLevelGGNN(Module):
    def __init__(
            self,
            annotation_size: int,
            num_layers: int,
            gate_nn: Module,
            hidden_size: int = 0,
            final_layer: Optional[Module] = None,
            nn: Optional[Module] = None,
            padding_mode: str = 'constant',
            padding_const: int = 0,
            graph_attention_impl: str = 'torch_geometric',
            ggnn_impl: str = 'torch_geometric',
            **kwargs,
    ):
        # Note: we define state_size = annotation_size + hidden_size;
        # therefore, we trivially have
        # state_size >= annotation_size (assuming unsigned integers).
        super(GraphLevelGGNN, self).__init__()
        self.annotation_size = annotation_size
        self.hidden_state = hidden_size
        self.padding_mode = padding_mode
        self.padding_const = padding_const
        self.ggnn_layer = make_ggnn(state_size=annotation_size + hidden_size,
                                    num_layers=num_layers, ggnn_impl=ggnn_impl,
                                    **kwargs)
        self.attention_layer = make_graph_attention(
            gate_nn=gate_nn, nn=nn, graph_attention_impl=graph_attention_impl,
            **kwargs)
        self.final_layer = final_layer

    def reset_parameters(self):
        self.ggnn_layer.reset_parameters()
        self.attention_layer.reset_parameters()

    def forward(self, x, edge_index, batch, **kwargs):
        # Step 1: pad `x` from `annotation_size` to `hidden_state +
        # annotation_size`
        assert x.shape[-1] == self.annotation_size
        x_ggnn = torch.nn.functional.pad(
            x, (0, self.hidden_state), self.padding_mode, self.padding_const)
        assert x_ggnn.shape[-1] == self.annotation_size + self.hidden_state

        # Step 2: pass the padded `x` into the GGNN layer
        x_ggnn = self.ggnn_layer(x, edge_index, **kwargs)

        # Step 3: catenate the GGNN output with the original input
        x = torch.cat((x_ggnn, x), -1)
        del x_ggnn
        assert x.shape[-1] == self.annotation_size + \
            self.hidden_state + self.annotation_size

        # Step 4: pass this through the attention layer
        x = self.attention_layer(x, batch)

        # Step 5: pass this through the final layer, if 
        if self.final_layer is not None:
            x = self.final_layer(x)

        return x

# TODO: change to more complicated nonlinear NN, etc...


def make_linear_gate_nn(annotation_size: int, hidden_state: int = 0) -> Module:
    return nn.Linear(2 * annotation_size + hidden_state, 1)
