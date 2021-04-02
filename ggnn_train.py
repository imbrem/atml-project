"""
RNN/LSTM baseline training for the bAbI dataset.

Adapted from Yujia Li,
https://github.com/yujiali/ggnn/blob/master/babi/babi_rnn_train.lua
https://github.com/yujiali/ggnn/blob/master/babi/run_rnn_baselines.py
"""

from data import BabiSequentialGraphDataset
from graph_level_ggnn import GraphLevelGGNN
from torch import nn
from torch.utils.data import DataLoader
import baseline_parameters
import torch
import argparse
import wandb
import os

wandb.init(project='ggsnn-rnn-baselines')

SEED = 8
N_THREADS = 1
N_FOLDS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_train_loaders(params, fold_id, n_train, dataset='sequential_graph'):
    if dataset == 'sequential_graph':
        train_dataset = BabiSequentialGraphDataset(params['root_dir'], fold_id,
                                                   params['task_id'],
                                                   params['n_targets'],
                                                   split='train',
                                                   n_train=n_train)
        val_dataset = BabiSequentialGraphDataset(params['root_dir'], fold_id,
                                                 params['task_id'],
                                                 params['n_targets'],
                                                 split='validation')

        train_loader = DataLoader(train_dataset,
                                  batch_size=params['batch_size'],
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=params['batch_size'],
                                shuffle=True)

        return train_loader, val_loader
    else:
        raise NotImplementedError('Other datasets not supported.')


def train(model, train_loader, val_loader, params, run_desc,
          patience=0, delta=0.005):
    """ Training procedure for the given number of iterations. """
    best_val_loss = None
    best_train_loss, best_train_acc, best_val_acc = 0., 0., 0.
    checkpoint = 'model_{}.pt'.format(run_desc)

    epochs = params['max_iters'] // params[
        'batch_size'] + 1 if patience == 0 else 0
    epoch = 0
    iters = 0
    while epoch < epochs or 0 <= iters < patience:
        model.train()
        train_loss = 0
        train_total, train_correct = 0., 0.
        # Training
        for data in train_loader:  # iterate through batches
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)

            loss = criterion(out.permute(0, 2, 1), data.y)
            loss.backward()

            train_loss += loss.item()
            train_correct += (out.argmax(dim=-1).eq(data.y)).all(dim=1).sum()
            train_total += len(data.y)

            # Gradient clipping as per original implementation.
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()

        mean_train_loss = train_loss / len(train_loader)
        assert (train_correct <= train_total)
        train_acc = train_correct / train_total

        # Validation
        with torch.set_grad_enabled(False):
            mean_val_loss, val_acc = evaluate(model, val_loader)

        wandb.log({'train_loss_{}'.format(run_desc): mean_train_loss,
                   'val_loss_{}'.format(run_desc): mean_val_loss,
                   'train_acc_{}'.format(run_desc): train_acc,
                   'val_acc_{}'.format(run_desc): val_acc})

        if best_val_loss is None or mean_val_loss < best_val_loss - delta:
            iters = 0
            torch.save(model.state_dict(), os.path.join(wandb.run.dir,
                                                        checkpoint))
            wandb.save(checkpoint)
            best_train_loss, best_val_loss = mean_train_loss, mean_val_loss
            best_train_acc, best_val_acc = train_acc, val_acc
            # print('{} train_loss_{}: {}'.format(epoch, run_desc,
            #                                     mean_train_loss))
            # print('{} val_loss_{}: {}'.format(epoch, run_desc, mean_val_loss))
            # print('{} train_acc_{}: {}'.format(epoch, run_desc, train_acc))
            # print('{} val_acc_{}: {}'.format(epoch, run_desc, val_acc))
        else:
            iters += 1
        epoch += 1

    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, checkpoint)))
    return model, [best_train_loss, best_val_loss, best_train_acc, best_val_acc]


def evaluate(model, loader):
    model.eval()
    total_loss = 0.
    total = 0.
    correct = 0.
    for sequences, targets in loader:
        sequences, targets = sequences.to(device), targets.to(device)

        outputs, _ = model(sequences)
        loss = criterion(outputs.permute(0, 2, 1), targets)
        total_loss += loss.item()
        correct += (outputs.argmax(dim=-1).eq(targets)).all(dim=1).sum()
        total += len(targets)

    mean_loss = total_loss / len(loader)
    acc = correct / total
    assert (correct <= total)
    return mean_loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, choices=[4, 15, 16, 18, 19])
    parser.add_argument('--all_data', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=250)
    args = parser.parse_args()

    task_id = args.task_id
    all_data = args.all_data
    patience = args.patience
    params = baseline_parameters.get_parameters_for_task(task_id)
    n_train_to_try = params['n_train_to_try'] if not all_data else [0]

    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)

    for n_train in n_train_to_try:
        fold_performances = []
        fold_test_performances = []
        for fold_id in range(1, N_FOLDS + 1):
            run_desc = '{}_fold_{}_n_train_{}'.format(model_type, fold_id,
                                                      n_train)
            model = None
            if params['n_targets'] == 1:
                model = GraphLevelGGNN(annotation_size=params['max_token_id'],
                                       num_layers=2,
                                       gate_nn=None,
                                       hidden_size=params['hidden_size'] -
                                                   params['max_token_id'],
                                       ggnn_impl='team2').to(device)
            else:
                raise NotImplementedError('ggsnn not supported')

            wandb.watch(model)
            wandb.config.update(params)
            wandb.log({'n_parameters': model.count_parameters()})
            wandb.run.name = 'task_{}_'.format(task_id) + run_desc

            train_loader, val_loader = get_train_loaders(params,
                                                         fold_id,
                                                         n_train)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=params['learning_rate'])
            criterion = nn.CrossEntropyLoss()

            # Train the model and obtain best train and validation performance
            model, fold_performance = train(model, train_loader, val_loader,
                                            params,
                                            run_desc,
                                            patience)

            # Logging train and validation performance for fold
            wandb.run.summary['final_train_loss_{}'.format(run_desc)] = \
                fold_performance[0]
            wandb.run.summary['final_val_loss_{}'.format(run_desc)] = \
                fold_performance[1]
            wandb.run.summary['final_train_acc_{}'.format(run_desc)] = \
                fold_performance[2]
            wandb.run.summary['final_val_acc_{}'.format(run_desc)] = \
                fold_performance[3]
            fold_performances.append(fold_performance)
            # print(
            #     'final_train_loss_{}: {}'.format(run_desc,
            #     fold_performance[0]))
            # print('final_val_loss_{}: {}'.format(run_desc,
            # fold_performance[1]))
            # print(
            #     'final_train_acc_{}: {}'.format(run_desc, fold_performance[
            #     2]))
            # print('final_val_acc_{}: {}\n'.format(run_desc, fold_performance[
            #     3]))

            test_loss, test_acc = evaluate(model, test_loader)
            wandb.run.summary['test_loss_{}'.format(run_desc)] = test_loss
            wandb.run.summary['test_acc_{}'.format(run_desc)] = test_acc
            fold_test_performances.append([test_loss, test_acc])
            print('test_loss_{}: {}'.format(run_desc, test_loss))
            print('test_acc_{}: {}\n'.format(run_desc, test_acc))

        final_performances = list(
            torch.tensor(fold_performances).mean(dim=0).numpy())
        wandb.run.summary['train_loss_{}'.format(n_train)] = \
            final_performances[0]
        wandb.run.summary['val_loss_{}'.format(n_train)] = \
            final_performances[1]
        wandb.run.summary['train_acc_{}'.format(n_train)] = \
            final_performances[2]
        wandb.run.summary['val_acc_{}'.format(n_train)] = \
            final_performances[3]
        # print(
        #     'train_loss_{}: {}'.format(n_train, final_performances[0]))
        # print(
        #     'val_loss_{}: {}'.format(n_train, final_performances[1]))
        # print(
        #     'train_acc_{}: {}'.format(n_train, final_performances[2]))
        # print(
        #     'val_acc_{}: {}\n'.format(n_train, final_performances[3]))

        final_test_means = list(torch.tensor(
            fold_test_performances).mean(dim=0).numpy())
        final_test_stds = list(torch.tensor(
            fold_test_performances).std(dim=0).numpy())
        wandb.run.summary['test_loss_mean_{}'.format(n_train)] = \
            final_test_means[0]
        wandb.run.summary['test_acc_mean_{}'.format(n_train)] = \
            final_test_means[1]
        wandb.run.summary['test_loss_std_{}'.format(n_train)] = \
            final_test_stds[0]
        wandb.run.summary['test_acc_std_{}'.format(n_train)] = \
            final_test_stds[1]

        print(
            'avg_test_loss_{}: {}'.format(n_train, final_test_means[0]))
        print(
            'avg_test_acc_{}: {}\n'.format(n_train, final_test_means[1]))
