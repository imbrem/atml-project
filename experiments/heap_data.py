import torch_geometric
import torch
import networkx as nx
from networkx import DiGraph
from typing import Optional, List, Any, Tuple
from torch_geometric.utils.convert import from_networkx
from heapq import heapify
import random
import torch.nn as nn

# Get the parent of an index into a heap


def parent(i: int) -> Optional[int]:
    if i <= 0:
        return None
    return (i - 1) // 2

# Check whether a list is a heap


def is_heap(l) -> bool:
    for (i, e) in enumerate(l):
        p = parent(i)
        if p is not None and l[p] > e:
            return False
    return True


MAX_LEN = 64

# Construct an array, with probability ~p_heapify of it being a heap, of length between min_len and max_len


def make_array(min_len: int = 1, max_len: int = MAX_LEN, p_heapify: float = 0.4) -> Tuple[List[int], bool]:
    result = []
    n = random.randint(min_len, max_len)
    for i in range(0, n):
        result.append(random.random())
    if random.random() < p_heapify:
        heapify(result)
        result_is_heap = True
    else:
        result_is_heap = is_heap(result)
    return (torch.tensor(result).view(n, -1), result_is_heap)

# Check whether a graph is that of a heap


def is_heap_graph(graph):
    for n, nbrs in graph.adj.items():
        for nbr in nbrs:
            if parent(nbr) != n:
                return False
    return True

# Make the graph of a heap


def make_heap_graph(n: int) -> DiGraph:
    result = DiGraph()
    result.add_nodes_from(range(0, n))
    result.add_edges_from(
        (parent(i), i) for i in range(0, n) if parent(i) is not None
    )
    assert is_heapgraph(result)
    return result

# Convert an array, which may be a heap, into a graph datapoint for a GNN


def nodes_to_gnn_datapoint(
    nodes,
    is_heap,
    min_len: int = 1,
    max_len: int = MAX_LEN,
    p_heap_graph: Optional[float] = None,
    and_y: bool = True
):
    n = nodes.shape[0]
    if p_heap_graph is not None or random.random() < p_heap_graph:
        graph = heapgraph(n)
        is_hg = True
    else:
        graph = nx.erdos_renyi_graph(n, 1/(n + 1), directed=True)
        is_hg = is_heapgraph(graph)

    data = from_networkx(graph)

    # Dumb edge case for when the number of edges is 0. Open an issue for torch_geometric?
    if data.edge_index.dtype != torch.long:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)

    assert data.edge_index.dtype == torch.long

    data.x = nodes
    if and_y:
        data.y = torch.tensor([float(is_heap and is_hg)])
    else:
        data.y = torch.tensor([
            float(is_heap),
            float(is_hg)
        ])

    return data

# Make a graph datapoint for a GNN


def make_gnn_datapoint(min_len: int = 1, max_len: int = MAX_LEN, p_heapify: float = 0.75, p_heap_graph: Optional[float] = None, and_y: bool = True) -> List[int]:
    (nodes, is_heap) = make_array(min_len, max_len, p_heapify)
    return nodes_to_gnn_datapoint(nodes, is_heap, min_len=min_len, max_len=max_len, p_heap_graph=p_heap_graph, and_y=and_y)

# Convert an array, which may be a heap, into a data point for an RNN


def nodes_to_rnn_datapoint(
    nodes,
    is_heap,
    min_len: int = 1,
    max_len: int = MAX_LEN,
    p_heap_graph: Optional[float] = None,
    and_y: bool = True
):
    raise NotImplementedError()