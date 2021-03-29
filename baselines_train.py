"""
RNN/LSTM baseline training for the bAbI dataset.

Adapted from Yujia Li,
https://github.com/yujiali/ggnn/blob/master/babi/babi_rnn_train.lua
"""

from baselines_data import BabiRNNDataset
from baselines.baseline_rnn import BaselineRNN
from torch import nn
from torch.utils.data import DataLoader
import rnn_parameters
import torch
import argparse

SEED = 8
N_THREADS = 1

model = 'rnn'
task_id = 4
n_train = 0
fold_id = 1

params = rnn_parameters.get_parameters_for_task(model, task_id)

# DATASET LOADERS
torch.manual_seed(SEED)
torch.set_num_threads(N_THREADS)

# TODO for every n_train_to_try
# for n_train in params['n_train_to_try']
# TODO for every fold

train_dataset = BabiRNNDataset(params['root_dir'], fold_id,
                               params['task_id'],
                               params['n_targets'], split='train',
                               n_train=n_train)
train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'],
                              shuffle=True)

val_dataset = BabiRNNDataset(params['root_dir'], fold_id,
                             params['task_id'],
                             params['n_targets'], split='validation')
val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'],
                            shuffle=True)

test_dataset = BabiRNNDataset(params['root_dir'], fold_id,
                              params['task_id'],
                              params['n_targets'], split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False)

# TODO set max_iters
# max_iters = n_epochs * math.ceil(n_train * 1. / batch_size)

# TODO better ways to obtain max_token_id
if model is 'rnn':
    model = BaselineRNN(input_size=train_dataset.max_token_id,
                        hidden_size=params['hidden_size'],
                        n_targets=params['n_targets'])
else:  # 'lstm'
    model = None

print('Total number of parameters: {}\n'.format(model.count_parameters()))

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
criterion = nn.CrossEntropyLoss()

# TODO output_directory
# TODO logging, early stopping
# print_every, save_every, plot_every

# Set up checkpoint directory
# Path(args.output_dir).mkdir(parents=True, exist_ok=True)
# print('Checkpoints will be saved to ', args.output_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_loader, val_loader):
    """ Training procedure for a single epoch. """
    model.train()

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

    return total_loss / len(train_loader), val_loss / len(val_loader)
