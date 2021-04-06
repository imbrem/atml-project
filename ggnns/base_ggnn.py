# A base GGNN layer
#
# Can optionally be configured just to use the torch_geometric implementation
from torch_geometric.typing import Adj, OptTensor
import torch
from torch import Tensor
import torch_geometric
from torch import nn
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import global_add_pool


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
        return torch_geometric.nn.GatedGraphConv(out_channels=state_size, num_layers=num_layers, aggr=aggr,
                                                 bias=bias, **kwargs)
    elif ggnn_impl == 'team2':
        return BaseGGNN(state_size=state_size, num_layers=num_layers, aggr=aggr, bias=bias, **kwargs)
    else:
        raise ValueError(f"Invalid GGNN implementation {ggnn_impl}")


# Adapted from the torch_geometric implementation to support edge_attr
class BaseGGNN(MessagePassing):
    def __init__(self, state_size: int, num_layers: int,
                 aggr: str = 'add',
                 bias: bool = True, total_edge_types: int = 4, **kwargs):
        super(BaseGGNN, self).__init__(aggr=aggr)

        self.state_size = state_size
        self.out_channels = state_size
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, state_size, state_size))
        self.rnn = torch.nn.GRUCell(state_size, state_size, bias=bias)

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
        return x_j if edge_attr is None else \
            torch.bmm(
                torch.einsum("ab,bcd->acd", (edge_attr, self.edge_type_weight)),
                x_j.unsqueeze(-1)).squeeze() + \
            self.edge_type_bias.repeat(x_j.size(0), 1)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)


class BaseNodeSelectionGGNN(nn.Module):
    def __init__(self, state_size: int, num_layers: int,
                 aggr: str = 'add',
                 bias: bool = True, total_edge_types: int = 4, **kwargs):
        super(BaseNodeSelectionGGNN, self).__init__()
        self.ggnn = BaseGGNN(state_size=state_size,
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
        return torch.unsqueeze(softmax(self.mlp(out), batch,
                                       num_nodes=size).view(size, -1), dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseGraphLevelGGNN(nn.Module):
    def __init__(self, state_size: int, num_layers: int,
                 aggr: str = 'add',
                 bias: bool = True, total_edge_types: int = 4,
                 annotation_size=1,
                 classification_categories=2, **kwargs):
        super(BaseGraphLevelGGNN, self).__init__()
        self.ggnn = BaseGGNN(state_size=state_size,
                             num_layers=num_layers, aggr=aggr,
                             bias=bias, total_edge_types=total_edge_types,
                             **kwargs)
        self.processing_net1 = nn.Sequential(
            nn.Linear(state_size + annotation_size,
                      2 * (state_size + annotation_size)),
            nn.ReLU(),
            nn.Linear(2 * (state_size + annotation_size),
                      state_size + annotation_size),
            nn.Sigmoid()
        )
        self.processing_net2 = nn.Sequential(
            nn.Linear(state_size + annotation_size,
                      2 * (state_size + annotation_size)),
            nn.ReLU(),
            nn.Linear(2 * (state_size + annotation_size),
                      state_size + annotation_size),
            nn.Tanh()
        )
        self.classification_layer = nn.Sequential(
            nn.Linear(state_size + annotation_size,
                      2 * classification_categories),
            nn.ReLU(inplace=True),
            nn.Linear(2 * classification_categories, classification_categories),
            nn.Softmax(dim=1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        out = self.ggnn(x, edge_index, edge_attr)
        out = torch.cat([out, x], dim=1)
        processed1 = self.processing_net1(out)
        processed2 = self.processing_net2(out)

        out = global_add_pool(processed1 * processed2, batch=batch)

        out = self.classification_layer(out)
        return torch.unsqueeze(out, 1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseGraphLevelGGSNN(nn.Module):
    def __init__(self, state_size: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, total_edge_types: int = 4,
                 annotation_size=1, pred_steps=2, classification_categories=4, **kwargs):
        super(BaseGraphLevelGGSNN, self).__init__()
        self.pred_steps = pred_steps
        self.ggnn = BaseGGNN(state_size=state_size, num_layers=num_layers, aggr=aggr,
                             bias=bias, total_edge_types=total_edge_types,
                             **kwargs)
        self.processing_net1 = nn.Sequential(
            nn.Linear(state_size + annotation_size,
                      2 * (state_size + annotation_size)),
            nn.ReLU(),
            nn.Linear(2 * (state_size + annotation_size),
                      state_size + annotation_size),
            nn.Sigmoid()
        )
        self.processing_net2 = nn.Sequential(
            nn.Linear(state_size + annotation_size,
                      2 * (state_size + annotation_size)),
            nn.ReLU(),
            nn.Linear(2 * (state_size + annotation_size),
                      state_size + annotation_size),
            nn.Tanh()
        )
        self.classification_layer = nn.Sequential(
            nn.Linear(state_size + annotation_size,
                      2 * classification_categories),
            nn.ReLU(inplace=True),
            nn.Linear(2 * classification_categories, classification_categories),
            nn.Softmax(dim=1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        outputs = list()
        latent = x
        for _ in range(self.pred_steps):
            latent = self.ggnn(latent, edge_index, edge_attr)
            out = torch.cat([latent, x], dim=1)
            processed1 = self.processing_net1(out)
            processed2 = self.processing_net2(out)
            out = global_add_pool(processed1 * processed2, batch=batch)
            out = self.classification_layer(out)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
