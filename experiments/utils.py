import networkx as nx
from torch_geometric.utils.convert import from_networkx

def from_networkx_fixed(G):
    """
    A fixed version of torch_geometric.utils.convert.from_networkx, which breaks for graphs with no edges. Should probably make a pull request or something...
    This is why dynamic types are bad... just saying...
    """
    data = from_networkx(graph)
    if data.edge_index.dtype != torch.long:
        assert data.edge_index.shape[-1] == 0
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
    return data

def directed_path_graph(n):
    """
    Create a directed path graph of length n

    A convenience constructor, mainly for ease of use to pass as a lambda and because it's too easy to just use `directed=True` by accident.
    """
    return nx.path_graph(n, create_using=nx.DiGraph)

def random_one_graph(n):
    """
    A graph for which, on average, every node is connected to one other node
    """
    return nx.fast_gnp_random_graph(n, 1/(n*n), directed=True)

def random_two_graph(n):
    """
    A graph for which, on average, every node is connected to two other nodes
    """
    return nx.fast_gnp_random_graph(n, 2/(n*n), directed=True)

def disconnected_graph(n):
    """
    A graph with no edges, and n nodes
    """
    g = nx.DiGraph()
    for i in range(0, n):
        g.add_node(i)
    return g