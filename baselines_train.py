"""
RNN/LSTM baseline training for the bAbI dataset.

Adapted from Yujia Li,
https://github.com/yujiali/ggnn/blob/master/babi/babi_rnn_train.lua
https://github.com/yujiali/ggnn/blob/master/babi/run_rnn_baselines.py
"""

from baselines_data import get_loaders
from baselines.baseline_rnn import BaselineRNN
from baselines.baseline_lstm import BaselineLSTM
from torch import nn
import baselines_parameters
import torch
import argparse
import wandb
import os

wandb.init(project='ggsnn-rnn-baselines')

SEED = 8
N_THREADS = 1
N_FOLDS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        train_total = 0.
        train_correct = 0.
        # Training
        for sequences, targets in train_loader:  # iterate through batches
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, _ = model(sequences)

            loss = criterion(outputs.permute(0, 2, 1), targets)
            loss.backward()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=-1).eq(targets)).all(
                dim=1).sum()
            train_total += len(targets)

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
    parser.add_argument('--model', type=str, help="RNN/LSTM", choices=['rnn',
                                                                       'lstm'])
    parser.add_argument('--all_data', type=bool, default=False)
    parser.add_argument('--use_embeddings', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=250)
    args = parser.parse_args()

    model_type = args.model
    task_id = args.task_id
    all_data = args.all_data
    patience = args.patience
    use_embeddings = args.use_embeddings
    params = baselines_parameters.get_parameters_for_task(model_type, task_id)
    n_train_to_try = params['n_train_to_try'] if not all_data else [0]

    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)

    for n_train in n_train_to_try:
        fold_performances = []
        fold_test_performances = []
        for fold_id in range(1, N_FOLDS + 1):
            model = None
            if model_type == 'rnn':
                model = BaselineRNN(input_size=params['max_token_id'],
                                    hidden_size=params['hidden_size'],
                                    n_targets=params['n_targets'],
                                    use_embeddings=use_embeddings)
            elif model_type == 'lstm':
                model = BaselineLSTM(input_size=params['max_token_id'],
                                     hidden_size=params['hidden_size'],
                                     n_targets=params['n_targets'],
                                     use_embeddings=use_embeddings)

            wandb.watch(model)
            wandb.config.update(params)
            wandb.log({'n_parameters': model.count_parameters()})

            run_desc = '{}_fold_{}_n_train_{}'.format(model_type, fold_id,
                                                      n_train)
            wandb.run.name = 'task_{}_'.format(task_id) + run_desc

            train_loader, val_loader, test_loader = get_loaders(params,
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
