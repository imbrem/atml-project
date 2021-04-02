# A base GGNN layer
#
# Can optionally be configured just to use the torch_geometric implementation
from torch_geometric.typing import Adj, OptTensor
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.inits import uniform


def make_ggnn(
        state_size: int,
        num_layers: int,
        aggr: str = 'add',
        bias: bool = True,
        ggnn_impl: str = 'torch_geometric',
        **kwargs
):
    """
    Make a GGNN having a given implementation
    """
    if ggnn_impl == 'torch_geometric':
        return torch_geometric.nn.GatedGraphConv(
            out_channels=state_size, num_layers=num_layers, aggr=aggr,
            bias=bias, **kwargs)
    elif ggnn_impl == 'team2':
        return BaseGGNN(
            out_channels=state_size, num_layers=num_layers, aggr=aggr,
            bias=bias,
            **kwargs)
    else:
        raise ValueError(f"Invalid GGNN implementation {ggnn_impl}")


# Adapted from the torch_geometric implementation to support edge_attr
class BaseGGNN(MessagePassing):
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super(BaseGGNN, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                edge_attr: OptTensor = None) -> Tensor:
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               edge_attr=edge_attr,
                               size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor,
                edge_attr: OptTensor):
        # TODO: edge_attr
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
