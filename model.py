import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F

class GGNN(nn.Module):
    def __init__(self, message_passing_layers=16, embedding_size=50):
        super(GGNN, self).__init__()
        self.message_passing_layers = nn.ModuleList([
            torch_geometric.nn.GCNConv(embedding_size, embedding_size) for _ in range(0, message_passing_layers)
        ])
        self.hidden_layer = torch.nn.Linear(embedding_size, embedding_size)
        self.final_layer = torch.nn.Linear(embedding_size, 1)

    def forward(self, x, edge_index, batch):
        for layer in self.message_passing_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)

        return x

    def reset_parameters(self):
        for layer in self.message_passing_layers:
            layer.reset_parameters()
        self.hidden_layer.reset_parameters()
        self.final_layer.reset_parameters()

class GGSNN(nn.Module):
    pass