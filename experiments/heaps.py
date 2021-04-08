import torch_geometric
import torch
import networkx as nx
import random
import torch.nn as nn

from experiments.utils import from_networkx_fixed
from networkx import DiGraph
from typing import Optional, List, Any, Tuple
from heapq import heapify

def parent(i: int) -> Optional[int]:
    """
    Get the index of a parent in an array representing a heap
    """
    if i <= 0:
        return None
    return (i - 1) // 2


def is_heap(l) -> bool:
    """
    Check whether a list is a heap
    """
    for (i, e) in enumerate(l):
        p = parent(i)
        if p is not None and l[p] > e:
            return False
    return True


MAX_LEN = 64


def make_array(min_len: int = 1, max_len: int = MAX_LEN, p_heapify: float = 0.4) -> Tuple[List[int], bool]:
    """
    Construct a random array of length between `min_len` and `max_len`, with probability `p_heapify` of being a heap.
    """
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


def is_heap_graph(graph):
    """
    Check whether a given graph is a heap with root node `0`.
    """
    for n, nbrs in graph.adj.items():
        for nbr in nbrs:
            if parent(nbr) != n:
                return False
    return True


def make_heap_graph(n: int) -> DiGraph:
    """
    Make a graph representing a heap with root node `0`; return a networkx digraoh
    """
    result = DiGraph()
    result.add_nodes_from(range(0, n))
    result.add_edges_from(
        (parent(i), i) for i in range(0, n) if parent(i) is not None
    )
    assert is_heapgraph(result)
    return result


def nodes_to_gnn_datapoint(
    nodes,
    is_heap,
    min_len: int = 1,
    max_len: int = MAX_LEN,
    p_heap_graph: Optional[float] = None,
    and_y: bool = True
):
    """
    Convert an array, which may be a heap, into a graph datapoint for a GNN
    """
    n = nodes.shape[0]
    if p_heap_graph is not None or random.random() < p_heap_graph:
        graph = heapgraph(n)
        is_hg = True
    else:
        graph = nx.erdos_renyi_graph(n, 1/(n + 1), directed=True)
        is_hg = is_heapgraph(graph)

    data = from_networkx_fixed(graph)

    data.x = nodes
    if and_y:
        data.y = torch.tensor([float(is_heap and is_hg)])
    else:
        data.y = torch.tensor([
            float(is_heap),
            float(is_hg)
        ])

    return data


def make_gnn_datapoint(min_len: int = 1, max_len: int = MAX_LEN, p_heapify: float = 0.75, p_heap_graph: Optional[float] = None, and_y: bool = True) -> List[int]:
    """
    Make a datapoint for a GNN
    """
    (nodes, is_heap) = make_array(min_len, max_len, p_heapify)
    return nodes_to_gnn_datapoint(nodes, is_heap, min_len=min_len, max_len=max_len, p_heap_graph=p_heap_graph, and_y=and_y)


def nodes_to_rnn_datapoint(
    nodes,
    is_heap,
    min_len: int = 1,
    max_len: int = MAX_LEN,
    p_heap_graph: Optional[float] = None,
    and_y: bool = True
):
    """
    Convert an array, which may be a heap, into a graph datapoint for an RNN
    """
    raise NotImplementedError()
