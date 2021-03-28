"""
RNN/LSTM baseline training for the bAbI dataset.

Adapted from Yujia Li,
https://github.com/yujiali/ggnn/blob/master/babi/babi_rnn_train.lua
"""

from baselines_data import BabiRNNDataset
from baselines.baseline_rnn import BaselineRNN
from torch import nn
from torch.utils.data import DataLoader
import torch
import argparse
from pathlib import Path
import math

# ARGUMENT PARSING

parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--model', default='rnn', type=str, help='rnn or lstm')
parser.add_argument('--embedding_size', default=50, type=int,
                    help='dimensionality of the embeddings')
parser.add_argument('--hidden_size', default=50, type=int,
                    help='dimensionality of the hidden layers')
parser.add_argument('--n_targets', default=1, type=int,
                    help='number of targets for each example, if > 1 the '
                         'targets will be treated as a sequence')

# Data parameters
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--task_id', default=4, type=int)
parser.add_argument('--n_train', default=0, type=int,
                    help='number of training instances, 0 to use all available')

# Training parameters
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='type of optimization algorithm to use')
parser.add_argument('--batch_size', default=10, type=int, help='minibatch size')
parser.add_argument('--n_epochs', default=0, type=int,
                    help='number of epochs to train, (overrides maxiters)')
parser.add_argument('--max_iters', default=1000, type=int,
                    help='maximum number of weight updates')
parser.add_argument('--n_threads', default=1, type=int,
                    help='set the number of threads to use with this process')

# Checkpoint and logging parameters
parser.add_argument('--output_dir', default='.', type=str,
                    help='output directory')
parser.add_argument('--print_every', default=10, type=int,
                    help='frequency of training information logs')
parser.add_argument('--save_every', default=100, type=int,
                    help='frequency of checkpoints')
parser.add_argument('--plot_every', default=10, type=int,
                    help='frequency of training curve updates')

parser.add_argument('--seed', default=8, type=int, help='random seed')

args = parser.parse_args()

# DATASET LOADERS
torch.manual_seed(args.seed)
torch.set_num_threads(args.n_threads)

train_dataset = BabiRNNDataset(args.root_dir, args.fold_id, args.task_id,
                               args.n_targets, split='train',
                               n_train=args.n_train)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = BabiRNNDataset(args.root_dir, args.fold_id, args.task_id,
                             args.n_targets, split='validation')
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

test_dataset = BabiRNNDataset(args.root_dir, args.fold_id, args.task_id,
                              args.n_targets, split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False)

if args.n_targets > 1:
    args.n_targets += 1  # add 1 to include the end symbol

if args.n_epochs > 0:
    args.max_iters = args.n_epochs * math.ceil(
        args.n_train * 1. / args.batch_size)

# TODO fix model initialisation
if args.model == 'rnn':
    model = BaselineRNN(input_size=train_dataset.max_token_id,
                        hidden_size=args.hidden_size,
                        output_size=train_dataset.max_token_id,
                        n_targets=args.n_targets)
elif args.model == 'lstm':
    model = None
else:
    parser.error('Unknown model type: {}'.format(args.model))
print('Total number of parameters: {}\n'.format(model.count_parameters()))

if args.optimizer is not 'adam':
    raise parser.error('Unsupported optimizer')
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = nn.CrossEntropyLoss()

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
