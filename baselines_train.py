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


def train(model, train_loader, val_loader, iters):
    """ Training procedure for the given number of iterations. """
    model.train()

    for _ in range(iters):
        total_loss = 0
        # Training
        for sequences, targets in train_loader:  # iterate through batches
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, _ = model(sequences)

            loss = criterion(outputs.permute(0, 2, 1), targets.argmax(dim=-1))
            loss.backward()

            total_loss += loss.item()

            # Gradient clipping as per original implementation.
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()

        # Validation
        with torch.set_grad_enabled(False):
            val_loss = 0
            for val_sequences, val_targets in val_loader:
                val_sequences, val_targets = val_sequences.to(
                    device), val_targets.to(
                    device)

                outputs, _ = model(val_sequences)
                loss = criterion(outputs.permute(0, 2, 1), val_targets.argmax(
                    dim=-1))
                val_loss += loss.item()

        print(total_loss / len(train_loader), val_loss / len(val_loader))

        # TODO output_directory
        # TODO logging, early stopping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', choices=[4, 15, 16, 18, 19])
    parser.add_argument('--model', help="RNN/LSTM", choices=['rnn', 'lstm'])
    args = parser.parse_args()

    model = args.model
    task_id = args.task_id
    params = baseline_parameters.get_parameters_for_task(model, task_id)
    n_train_to_try = params['n_train_to_try']

    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)

    for n_train in n_train_to_try:
        for fold_id in range(1, N_FOLDS + 1):
            train_loader, val_loader, test_loader = get_loaders(params,
                                                                fold_id,
                                                                n_train)
            if model is 'rnn':
                model = BaselineRNN(input_size=params['max_token_id'],
                                    hidden_size=params['hidden_size'],
                                    n_targets=params['n_targets'])
            else:  # 'lstm'
                model = BaselineLSTM(input_size=params['max_token_id'],
                                     hidden_size=params['hidden_size'],
                                     n_targets=params['n_targets'])

            print('Total number of parameters: {}\n'.format(
                model.count_parameters()))

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=params['learning_rate'])
            criterion = nn.CrossEntropyLoss()

            iters = params['max_iters'] / params['batch_size']

            # TODO Set up checkpoint directory
            # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            # print('Checkpoints will be saved to ', args.output_dir)
            train(model, train_loader, val_loader, iters)

