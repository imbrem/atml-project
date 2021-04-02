# An implementation of global soft attention
import torch_geometric
from torch.nn import Module
from typing import Union, List, Optional


def make_graph_attention(
    gate_nn: Module,
    nn: Optional[Module] = None,
    graph_attention_impl='torch_geometric',
    **kwargs
):
    if graph_attention_impl == 'torch_geometric':
        return torch_geometric.nn.GlobalAttention(gate_nn=gate_nn, nn=nn)
    elif graph_attention_impl == 'team2':
        raise NotImplementedError("Our graph attention")
    else:
        raise ValueError(
            f"Invalid graph attention implementation {graph_attention_impl}")
