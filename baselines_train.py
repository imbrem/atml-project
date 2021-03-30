"""
RNN/LSTM baseline training for the bAbI dataset.

Adapted from Yujia Li,
https://github.com/yujiali/ggnn/blob/master/babi/babi_rnn_train.lua
https://github.com/yujiali/ggnn/blob/master/babi/run_rnn_baselines.py
"""

from baselines_data import BabiRNNDataset
from baselines.baseline_rnn import BaselineRNN
from baselines.baseline_lstm import BaselineLSTM
from torch import nn
from torch.utils.data import DataLoader
import baseline_parameters
import torch
import argparse
import wandb
from datetime import datetime

wandb.init(project='ggsnn-rnn-baselines')

SEED = 8
N_THREADS = 1
N_FOLDS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_loaders(params, fold_id, n_train):
    train_dataset = BabiRNNDataset(params['root_dir'], fold_id,
                                   params['task_id'],
                                   params['n_targets'], split='train',
                                   n_train=n_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=params['batch_size'],
                              shuffle=True)

    val_dataset = BabiRNNDataset(params['root_dir'], fold_id,
                                 params['task_id'],
                                 params['n_targets'],
                                 split='validation')
    val_loader = DataLoader(val_dataset,
                            batch_size=params['batch_size'],
                            shuffle=True)

    test_dataset = BabiRNNDataset(params['root_dir'], fold_id,
                                  params['task_id'],
                                  params['n_targets'], split='test')
    test_loader = DataLoader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, iters, run_desc):
    """ Training procedure for the given number of iterations. """
    mean_train_loss, mean_val_loss, train_acc, val_acc = 0., 0., 0., 0.

    model.train()

    epoch = 0
    for _ in range(iters):
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
            train_correct += (outputs.argmax(dim=-1).eq(targets)).sum()
            train_total += len(targets)

            # Gradient clipping as per original implementation.
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()

        # Validation
        with torch.set_grad_enabled(False):
            val_loss = 0
            val_total = 0.
            val_correct = 0.
            for val_sequences, val_targets in val_loader:
                val_sequences, val_targets = val_sequences.to(
                    device), val_targets.to(
                    device)

                outputs, _ = model(val_sequences)
                loss = criterion(outputs.permute(0, 2, 1), val_targets)
                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=-1).eq(val_targets)).sum()
                val_total += len(val_targets)

        mean_train_loss = train_loss / len(train_loader)
        mean_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        wandb.log({'train_loss_{}'.format(run_desc): mean_train_loss,
                   'val_loss_{}'.format(run_desc): mean_val_loss,
                   'train_acc_{}'.format(run_desc): train_acc,
                   'val_acc_{}'.format(run_desc): val_acc})

        epoch += 1

    wandb.save('model_{}.h5'.format(run_desc))
    return mean_train_loss, mean_val_loss, train_acc, val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', choices=[4, 15, 16, 18, 19])
    parser.add_argument('--model', help="RNN/LSTM", choices=['rnn', 'lstm'])
    args = parser.parse_args()

    model_type = args.model
    task_id = args.task_id
    params = baseline_parameters.get_parameters_for_task(model_type, task_id)
    n_train_to_try = params['n_train_to_try']

    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)

    for n_train in n_train_to_try:
        fold_performances = []
        for fold_id in range(1, N_FOLDS + 1):
            if model_type is 'rnn':
                model = BaselineRNN(input_size=params['max_token_id'],
                                    hidden_size=params['hidden_size'],
                                    n_targets=params['n_targets'])
            else:  # 'lstm'
                model = BaselineLSTM(input_size=params['max_token_id'],
                                     hidden_size=params['hidden_size'],
                                     n_targets=params['n_targets'])

            wandb.run.save()
            wandb.watch(model)
            wandb.config.update(params)
            wandb.log({'n_parameters': model.count_parameters()})

            run_desc = 'fold_{}_n_train_{}'.format(fold_id, n_train)
            wandb.run.name = '{}_task_{}'.format(model_type, task_id) + run_desc

            wandb.run.save()
            early_stopping_checkpoint = '{}_{}_state_dict.pt'.format(
                datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                wandb.run.name)

            train_loader, val_loader, test_loader = get_loaders(params,
                                                                fold_id,
                                                                n_train)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=params['learning_rate'])
            criterion = nn.CrossEntropyLoss()

            iters = params['max_iters'] / params['batch_size']
            fold_performance = train(model, train_loader, val_loader, iters,
                                     run_desc)
            wandb.log({'final_train_loss_{}'.format(run_desc):
                           fold_performance[0],
                       'final_val_loss_{}'.format(run_desc):
                           fold_performance[1],
                       'final_train_acc_{}'.format(run_desc):
                           fold_performance[2],
                       'final_val_acc_{}'.format(run_desc):
                           fold_performance[3]})

            fold_performances.append(fold_performance)

        final_performances = list(
            torch.tensor(fold_performances).mean(dim=0).numpy())
        wandb.log({"final_train_loss": fold_performances[0],
                   "final_val_loss": fold_performances[1],
                   "final_train_acc": fold_performances[2],
                   "final_val_acc": fold_performances[3]})
