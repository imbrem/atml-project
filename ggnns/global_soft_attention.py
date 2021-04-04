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
    """Global soft attention layer.

    Args:
        gate_nn (torch.nn.Module): A neural network that computes attention
            scores by mapping node features `x` of shape `[-1, in_channels]`
            to shape `[-1, 1]`.
        nn (torch.nn.Module, optional): A neural network that maps node
            features `x` of shape `[-1, in_channels]` to shape :`[-1,
            out_channels]` before combining them with the attention scores.
    """
    if graph_attention_impl == 'torch_geometric':
        return torch_geometric.nn.GlobalAttention(gate_nn=gate_nn, nn=nn)
    elif graph_attention_impl == 'team2':
        raise NotImplementedError("Our graph attention")
    else:
        raise ValueError(
            f"Invalid graph attention implementation {graph_attention_impl}")
