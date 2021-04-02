# A base GGNN layer
#
# Can optionally be configured just to use the torch_geometric implementation
from torch_geometric.typing import Adj, OptTensor
import torch
from torch import Tensor
import torch_geometric
from torch import nn
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.inits import glorot


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
            state_size=state_size, num_layers=num_layers, aggr=aggr, bias=bias,
            **kwargs)
    else:
        raise ValueError(f"Invalid GGNN implementation {ggnn_impl}")


# Adapted from the torch_geometric implementation to support edge_attr
class BaseGGNN(MessagePassing):
    def __init__(self, state_size: int, out_channels: int, num_layers: int,
                 aggr: str = 'add',
                 bias: bool = True, total_edge_types: int = 4, **kwargs):
        super(BaseGGNN, self).__init__(aggr=aggr)

        self.state_size = state_size
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(state_size, out_channels, bias=bias)

        # edge_type_tensor should be of the type (e, D, D), where e is the
        # total number of edge types
        # and D is the feature size
        self.edge_type_weight = Param(
            torch.zeros(total_edge_types, state_size, state_size),
            requires_grad=True)
        self.edge_type_bias = Param(torch.zeros(1, state_size),
                                    requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_type_weight)
        glorot(self.edge_type_bias)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """
        edge_attr: tensor of size (n, e) - n is the number of edges, e is the
        total number of edge types
        """
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m,
                               edge_attr=edge_attr,
                               size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_attr: OptTensor):
        """
        edge_attr: tensor of size (n, e) - n is the number of edges, e is the
        total number of edge types
        """
        # print(torch.bmm(torch.einsum("ab,bcd->acd", (edge_attr,
        # self.edge_type_weight)), x_j.unsqueeze(-1)).size())
        return x_j if edge_attr is None else \
            torch.bmm(
                torch.einsum("ab,bcd->acd", (edge_attr, self.edge_type_weight)),
                x_j.unsqueeze(-1)).squeeze() + \
            self.edge_type_bias.repeat(x_j.size(0), 1)

    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)


class BaseNodeSelectionGGNN(nn.Module):
    def __init__(self, state_size: int, out_channels: int, num_layers: int,
                 aggr: str = 'add',
                 bias: bool = True, total_edge_types: int = 4, **kwargs):
        super(BaseNodeSelectionGGNN, self).__init__()
        self.ggnn = BaseGGNN(state_size=state_size, out_channels=out_channels,
                             num_layers=num_layers, aggr=aggr,
                             bias=bias, total_edge_types=total_edge_types,
                             **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(state_size, state_size),
            nn.ReLU(inplace=True),
            nn.Linear(state_size, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        size = batch[-1].item() + 1
        out = self.ggnn(x, edge_index, edge_attr)
        return softmax(self.mlp(out), batch, num_nodes=size).view(size, -1)
