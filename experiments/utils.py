import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from tqdm import trange

def from_networkx_fixed(G):
    """
    A fixed version of torch_geometric.utils.convert.from_networkx, which breaks for graphs with no edges. Should probably make a pull request or something...
    This is why dynamic types are bad... just saying...
    """
    data = from_networkx(G)
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


def do_epoch(
    model,
    data,
    criterion,
    checker=None,
    opt=None,
    prefix="",
):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_correct = 0
    epoch_total = 0
    count = 0
    for batch in data:
        count += 1
        out = model(batch)
        loss = criterion(out, batch)
        loss.backward()
        epoch_loss += float(loss)
        if checker is not None:
            correct, total = checker(out, batch)
            assert total >= correct
            epoch_correct += int(correct)
            epoch_total += int(total)
        if opt is not None:
            opt.step()
            opt.zero_grad()
    result = {
        f"{prefix}loss": epoch_loss / count,
    }
    if checker is not None:
        result[f"{prefix}accuracy"] = epoch_correct / epoch_total
    return result


def train(
    model,
    opt,
    criterion,
    training_data,
    checker=None,
    extra_criteria=[],
    epochs=1,
    testing_data=None,
    logger=None
):
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    for epoch in trange(0, epochs):
        #model.train()
        train_results = do_epoch(
            model=model,
            data=training_data,
            criterion=criterion,
            checker=checker,
            opt=opt,
            prefix="train_")
        if logger is not None:
            logger.log(
                train_results
            )
        train_loss.append(train_results["train_loss"])
        if checker is not None:
            train_accuracy.append(train_results["train_accuracy"])

        if testing_data is not None:
            #model.test()
            test_results = do_epoch(
                model=model,
                data=testing_data,
                criterion=criterion,
                checker=checker,
                opt=None,
                prefix="test_")
            if logger is not None:
                logger.log(
                    test_results
                )
            test_loss.append(test_results["test_loss"])
            if checker is not None:
                test_accuracy.append(test_results["test_accuracy"])

        opt.zero_grad()
    return {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
