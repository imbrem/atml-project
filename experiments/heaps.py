import torch_geometric
import torch
import networkx as nx
import random
import torch.nn as nn

from experiments.utils import from_networkx_fixed, disconnected_graph
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


def maybe_make_heap(p_heap: float = 0.5, min_len: int = 1, max_len: int = MAX_LEN) -> Tuple[List[int], bool]:
    """
    Construct a random array of length between `min_len` and `max_len`, with probability `p_heap` of being a heap.
    """
    result = []
    n = random.randint(min_len, max_len)
    for i in range(0, n):
        result.append(random.random())
    if random.random() < p_heap:
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


def heap_graph(n: int) -> DiGraph:
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


def make_heap_test_gnn_datapoints(
    n,
    p_heap: float = 0.5,
    min_len: int = 1,
    max_len: int = MAX_LEN,
    graph_generators: List[Any] = [(heap_graph, True)],
    graph_probabilities: List[Any] = None,
    categorize: bool = False
):
    """
    Generate n graph datapoints for a GNN for the heap testing problem.

    `graph_generators` is a list of functions taking in a node count and returning a graph, and whether
    a heap paired with such a graph should be categorized as a heap, or `None`, 
    in which case a heap graph will always be used. `graph_probabilities` is a weighted list of
    probabilities a given graph will be chosen; if not proveded, a generator will be chosen at random.

    `categorize` determines whether to one-hot encode the chosen graph generator as an additional set of data points to predict.
    """
    data_list = []
    no_generators = len(graph_generators)
    for i, (gf, g_heap_graph) in random.choices(enumerate(graph_generators), weights=graph_probabilities):
        nodes, is_heap = maybe_make_heap()
        n = nodes.shape[-1]
        g = gf(n)
        g_is_heap = is_heap_graph(g)
        data = from_networkx_fixed(g(n))
        if categorize:
            data.y = torch.zeros((no_generators + 1))
            data.y[i] = 1.0
        data.y[-1] = int(is_heap_graph and is_heap)
        data_list.append(data)

    return data_list

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
