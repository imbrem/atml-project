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