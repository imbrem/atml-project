import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric 
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from model import GlobalGGNN

def generate_split_cycles(n):
    for s in range(3, n - 2):
        left = nx.cycle_graph(s)
        right = nx.cycle_graph(n - s)
        yield nx.disjoint_union(left, right)

def generate_cycles(n):
    cycle = nx.cycle_graph(n)
    for _ in range(3, n - 2):
        yield cycle

def convert_graph(graph, label):
    data = torch_geometric.utils.convert.from_networkx(graph)
    data.x = torch.randn(data.num_nodes, 50)
    data.y = label
    return data

def train_test_split_indices(n, k):
    s = n // k
    permutation = np.random.permutation(n)
    for b in range(0, n, s):
        test_permutation = permutation[b:b+s]
        train_permutation = np.concatenate((permutation[0:b], permutation[b+s:]))
        yield (train_permutation, test_permutation)

def train_test_splits(split_cycles, cycles, k, batch_size):
    n = len(split_cycles)
    assert n == len(cycles)
    for (train, test) in train_test_split_indices(n, k):
        train_data = [split_cycles[i] for i in train] + [cycles[i] for i in train]
        train_loader = torch_geometric.data.DataLoader(train_data, batch_size, shuffle=True)
        test_data = [split_cycles[i] for i in test] + [cycles[i] for i in test]
        test_loader = torch_geometric.data.DataLoader(test_data, batch_size, shuffle=True)
        yield (
            train_loader,
            test_loader
        )

def train(model, data_loader, optimizer, epochs=200):
    model.train()
    criterion = torch.nn.BCELoss()
    losses = []
    for epoch in trange(0, epochs):
        epoch_loss = 0.0
        count = 0
        for data in data_loader:
            out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())
            y = torch.reshape(data.y, [-1, 1]).cuda()
            loss = criterion(out, y)
            loss.backward()
            epoch_loss += float(loss)
            count += 1
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss /= count
        #print(f"Epoch {epoch} loss = {epoch_loss}")
        losses.append(epoch_loss)
    return losses

def single_test(model, data_loader):
    model.eval()
    correct = 0
    for data in data_loader:
        out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())
        pred = out.view([-1]).round()
        correct += int((pred == data.y.cuda()).sum())
    return correct / len(data_loader.dataset) 


def test(model, data_loader, k=10):
    test_results = []
    for _ in range(k):
        test_results.append(single_test(model, data_loader))
    return test_results

def validate(model, optimizer, split_cycles, cycles, k, batch_size, epochs=200):
    test_accs = []
    losses = []
    for (i, (train_data, test_data)) in enumerate(train_test_splits(split_cycles, cycles, k, batch_size)):
        model.reset_parameters()
        loss = train(model, train_data, optimizer, epochs)
        test_acc = test(model, test_data)
        std = np.std(test_acc)
        avg = np.mean(test_acc)
        print(f"\nSplit {i}/{k} accuracies = {test_acc} (avg = {avg}, std = {std})\n", flush=True)
        losses.append(loss)
        test_accs.append(avg)
    return (losses, test_accs)

if __name__ == "__main__":
    split_cycles = [convert_graph(g, 0.0) for n in range(6, 16) for g in generate_split_cycles(n)]
    cycles = [convert_graph(g, 1.0) for n in range(6, 16) for g in generate_cycles(n)]
    ggnn = GlobalGGNN(50, 1).cuda()
    optimizer = torch.optim.Adam(ggnn.parameters(), lr=0.01)
    losses, accuracy = validate(ggnn, optimizer, split_cycles, cycles, 5, 88)
    print(f"\n\nSplit accuracies: {accuracy}")
