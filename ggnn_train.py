"""
GGNN training code.

Adapted from Yujia Li,
# TODO ggnn directories
"""

from ggnn_data import get_data_loaders, get_n_edge_types
from ggnns.base_ggnn import BaseGraphLevelGGNN, BaseNodeSelectionGGNN, BaseGraphLevelGGSNN
from torch import nn
import ggnn_parameters
import torch
import wandb
import os
import argparse

SEED = 8
N_THREADS = 1
N_FOLDS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_loader, val_loader, model, optimizer,
          criterion, params,
          run_desc, patience=0,
          delta=0.005, log=True):
    """ Training procedure for the given number of iterations. """
    best_val_loss = None
    best_train_loss, best_train_acc, best_val_acc = 0., 0., 0.
    checkpoint = 'model_{}.pt'.format(run_desc)

    epochs = params['max_iters'] // params[
        'batch_size'] + 1 if patience == 0 else 0
    epoch = 0
    iters = 0
    while epoch < epochs or 0 <= iters < patience:
        train_loss, train_accuracy = train_epoch(train_loader, model,
                                                 optimizer, criterion)

        if log:
            wandb.log({'train_loss_{}'.format(run_desc): train_loss,
                       'train_acc_{}'.format(run_desc): train_accuracy})

        # Validation
        val_loss, val_accuracy = evaluate(val_loader, model, criterion)
        if log:
            wandb.log({'val_loss_{}'.format(run_desc): val_loss,
                       'val_acc_{}'.format(run_desc): val_accuracy})

        # if best_val_loss is None or val_loss < best_val_loss - delta:
        if best_val_loss is None or val_loss < best_val_loss:
            iters = 0
            if log:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir,
                                                            checkpoint))
                wandb.save(checkpoint)
            best_train_loss, best_val_loss = train_loss, val_loss
            best_train_acc, best_val_acc = train_accuracy, val_accuracy
        else:
            iters += 1
        epoch += 1

    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, checkpoint)))
    return model, [best_train_loss, best_val_loss, best_train_acc, best_val_acc]


def train_epoch(train_loader, model, optimizer, criterion):
    model.train(),
    total_loss = 0
    total_correct, total_examples = 0., 0.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch)
        loss = criterion(out.permute(0, 2, 1), data.y)
        loss.backward()
        optimizer.step()

        examples = data.y.size(0)
        total_loss += loss.item() * examples
        total_examples += examples
        total_correct += (out.argmax(dim=-1).eq(data.y)).all(dim=1).sum()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(loader, model, criterion):
    model.eval()
    total_loss = 0.
    total_examples = 0.
    total_correct = 0.
    for data in loader:
        data = data.to(device)
        out = model(x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch)
        loss = criterion(out.permute(0, 2, 1), data.y)

        examples = data.y.size(0)
        total_loss += loss.item() * examples
        total_examples += examples
        total_correct += (out.argmax(dim=-1).eq(data.y)).all(dim=1).sum()

    return total_loss / total_examples, total_correct / total_examples


def run_experiment(task_id, dataset='babi_graph', all_data=False, patience=250,
                   log=True):
    if log:
        wandb.init(project='ggsnn')
    params = ggnn_parameters.get_parameters_for_task(task_id)
    n_train_to_try = params['n_train_to_try'] if not all_data else [0]

    if dataset == 'sequential_graph':
        params['state_size'] = params['max_token_id']
        params['annotation_size'] = params['max_token_id']

    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)

    for n_train in n_train_to_try:
        fold_performances = []
        fold_test_performances = []
        for fold_id in range(1, N_FOLDS + 1):
            train_loader, val_loader, test_loader = get_data_loaders(params,
                                                                     fold_id,
                                                                     n_train,
                                                                     dataset)
            n_edge_types = get_n_edge_types(params, task_id)

            run_desc = 'ggnn_fold_{}_n_train_{}'.format(fold_id, n_train)
            if params['mode'] == 'graph_level':
                model = BaseGraphLevelGGNN(
                    state_size=params['state_size'],
                    num_layers=params['n_layers'],
                    total_edge_types=n_edge_types,
                    annotation_size=params['annotation_size'],
                    classification_categories=2).to(
                    device)
            elif params['mode'] == 'node_level':
                model = BaseNodeSelectionGGNN(
                    state_size=params['state_size'],
                    num_layers=params['n_layers'],
                    total_edge_types=n_edge_types).to(
                    device)
            elif params['mode'] == 'seq_graph_level':
                raise NotImplementedError()
            elif params['mode'] == 'share_seq_graph_level':
                model = BaseGraphLevelGGSNN(
                    state_size=params['state_size'],
                    num_layers=params['n_layers'],
                    total_edge_types=n_edge_types,
                    annotation_size=params['annotation_size']
                ).to(device)
            elif params['mode'] == 'share_seq_node_level':
                raise NotImplementedError()
            else:
                raise NotImplementedError()

            if log:
                wandb.watch(model)
                wandb.config.update(params)
                wandb.log({'n_parameters': model.count_parameters()})
                wandb.run.name = 'task_{}_'.format(task_id) + run_desc

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=params['learning_rate'])
            criterion = nn.CrossEntropyLoss()

            # Train the model and obtain best train and validation performance
            model, fold_performance = train(train_loader, val_loader, model,
                                            optimizer, criterion,
                                            params, run_desc, patience, log=log)

            if log:
                # Logging train and validation performance for fold
                wandb.run.summary['train_loss_{}'.format(run_desc)] = \
                    fold_performance[0]
                wandb.run.summary['val_loss_{}'.format(run_desc)] = \
                    fold_performance[1]
                wandb.run.summary['train_acc_{}'.format(run_desc)] = \
                    fold_performance[2]
                wandb.run.summary['val_acc_{}'.format(run_desc)] = \
                    fold_performance[3]
                fold_performances.append(fold_performance)

            test_loss, test_acc = evaluate(test_loader, model, criterion)
            if log:
                wandb.run.summary['test_loss_{}'.format(run_desc)] = test_loss
                wandb.run.summary['test_acc_{}'.format(run_desc)] = test_acc
            fold_test_performances.append([test_loss, test_acc])
            print('test_loss_{}: {}'.format(run_desc, test_loss))
            print('test_acc_{}: {}\n'.format(run_desc, test_acc))

        final_performances = list(
            torch.tensor(fold_performances).mean(dim=0).numpy())
        if log:
            wandb.run.summary['final_train_loss_{}'.format(n_train)] = \
                final_performances[0]
            wandb.run.summary['final_val_loss_{}'.format(n_train)] = \
                final_performances[1]
            wandb.run.summary['final_train_acc_{}'.format(n_train)] = \
                final_performances[2]
            wandb.run.summary['final_val_acc_{}'.format(n_train)] = \
                final_performances[3]

        final_test_means = list(torch.tensor(
            fold_test_performances).mean(dim=0).numpy())
        final_test_stds = list(torch.tensor(
            fold_test_performances).std(dim=0).numpy())
        if log:
            wandb.run.summary['test_loss_mean_{}'.format(n_train)] = \
                final_test_means[0]
            wandb.run.summary['test_acc_mean_{}'.format(n_train)] = \
                final_test_means[1]
            wandb.run.summary['test_loss_std_{}'.format(n_train)] = \
                final_test_stds[0]
            wandb.run.summary['test_acc_std_{}'.format(n_train)] = \
                final_test_stds[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experimental settings.')
    parser.add_argument('--task_id', '-ti', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='babi_graph',
                        choices=['babi_graph', 'sequential_graph'])
    parser.add_argument('--all_data', type=bool, default=False)
    parser.add_argument('--log', '-log', type=bool, default=True)
    args = parser.parse_args()
    run_experiment(task_id=args.task_id, log=args.log, dataset=args.dataset)
