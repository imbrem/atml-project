import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric 
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from tqdm import trange

def generate_split_cycles(n):
    for s in range(3, n - 2):
        left = nx.cycle_graph(s)
        right = nx.cycle_graph(n - s)
        yield nx.disjoint_union(left, right)

def generate_cycles(n):
    cycle = nx.cycle_graph(n)
    for _ in range(3, n - 2):
        yield cycle

def convert_graph(graph, label, no_attributes=2):
    data = torch_geometric.utils.convert.from_networkx(graph)
    data.x = torch.randn(data.num_nodes, no_attributes)
    data.y = label
    return data

def id_graph(graph, no_attributes=2):
    data = torch_geometric.utils.convert.from_networkx(graph)
    # For ReLU purposes...
    data.x = torch.randn(data.num_nodes, no_attributes).abs()
    data.y = data.x
    return data

def mod_node_graph(graph, mod_edge_no=2, no_attributes=2):
    data = torch_geometric.utils.convert.from_networkx(graph)
    data.x = torch.randn(data.num_nodes, no_attributes)
    data.y = len(graph.edges) % mod_edge_no
    return data

def mod_node_cycles(n, mod_edge_no=2, no_attributes=2):
    for k in range(6, n + 1):
        for g in generate_cycles(k):
            yield mod_node_graph(g, mod_edge_no, no_attributes)

def train(model, data_loader, optimizer, criterion, epochs=200):
    model.train()
    losses = []
    for epoch in trange(0, epochs):
        epoch_loss = 0.0
        count = 0
        for data in data_loader:
            out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())
            loss = criterion(out, data.y.cuda())
            loss.backward()
            epoch_loss += float(loss)
            count += 1
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss /= count
        #print(f"Epoch {epoch} loss = {epoch_loss}")
        losses.append(epoch_loss)
    return losses