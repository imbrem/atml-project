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

# autopep8: off
parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--model', default='rnn', type=str, help='rnn or lstm')
parser.add_argument('--embedding_size', default=50, type=int, help='dimensionality of the embeddings')
parser.add_argument('--hidden_size', default=50, type=int, help='dimensionality of the hidden layers')
parser.add_argument('--n_targets', default=1, type=int, help='number of targets for each example, if > 1 the targets will be treated as a sequence')

# Data parameters
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--task_id', default=4, type=int)
parser.add_argument('--n_train', default=0, type=int, help='number of training instances, 0 to use all available')
parser.add_argument('--n_val', default=50, type=int, help='number of validation instances (will not be used if datafile.val exists)')

# Training parameters
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--optimizer', default='adam', type=str, help='type of optimization algorithm to use')
parser.add_argument('--batch_size', default=10, type=int, help='minibatch size')
parser.add_argument('--n_epochs', default=0, type=int, help='number of epochs to train, (overrides maxiters)')
parser.add_argument('--max_iters', default=1000, type=int, help='maximum number of weight updates')
parser.add_argument('--n_threads', default=1, type=int, help='set the number of threads to use with this process')

# Checkpoint and logging parameters
parser.add_argument('--output_dir', default='.', type=str, help='output directory')
parser.add_argument('--print_every', default=10, type=int, help='frequency of training information logs')
parser.add_argument('--save_every', default=100, type=int, help='frequency of checkpoints')
parser.add_argument('--plot_every', default=10, type=int, help='frequency of training curve updates')

parser.add_argument('--seed', default=8, type=int, help='random seed')

args = parser.parse_args()
# autopep8: on

# Set up checkpoint directory
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
print('Checkpoints will be saved to ', args.output_dir)


train_dataset = BabiRNNDataset(args.root_dir, args.task_id, args.n_targets)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = BabiRNNDataset(args.root_dir, args.task_id, args.n_targets, validation=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

test_dataset = BabiRNNDataset(args.root_dir, args.task_id, args.n_targets, test=True)
test_dataloader = DataLoader(test_dataset, shuffle=False)

if args.n_targets > 1:
    args.n_targets += 1  # add 1 to include the end symbol

torch.set_num_threads(args.n_threads)

if args.n_epochs > 0:
    args.max_iters = args.n_epochs * math.ceil(args.n_train * 1. / args.batch_size)


# PREPARE DATA

torch.manual_seed(args.seed)

# TODO separate printing

print('Number of output classes: {}\n'.format(output_size))
print('Total number of weight updates: {}\n'.format(args.max_iters))

# TODO fix model initialisation
if args.model == 'rnn':
    model = BaselineRNN(input_size=args.input_size, hidden_size=args.hidden_size)
elif args.model == 'lstm':
    model = None
else:
    parser.error('Unknown model type: {}'.format(args.model))

if args.optimizer is not 'adam':
    raise parser.error('Unsupported optimizer')
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Total number of parameters: {}\n'.format(model.count_parameters()))

# TODO logging, validation loss, early stopping
# print_every, save_every, plot_every


def train(train_loader):
    """ Training procedure for a single epoch. """
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # if args.n_targets > 1:
        #         t_batch = torch.reshape(t_batch, t_batch.size(
        #             0) * args.n_targets, t_batch.size(1) / args.n_targets)

        out = model(data)

        # TODO what if sequence output
        loss = criterion(out, data.y)
        loss.backward()

        total_loss += loss.item()

        # Gradient clipping as per original implementation.
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

    return total_loss / len(train_loader)


# TODO error evaluation
# def eval_error(model, x, t):
#     pred = model.predict(x, args.n_targets)
#     # pred = pred.typeAs(t)
#     if args.n_targets > 1:
#         # TODO wth
#         return pred.ne(t).type('torch.DoubleTensor').sum(1).gt(0).type('torch.DoubleTensor').mean()
#     else:
#         # TODO fix
#         return pred.ne(t).type('torch.DoubleTensor').mean()

# TODO main
# if __name__ == "__main__":
#     from torch_geometric.data import DataLoader

#     dataroot = 'babi_data/processed_1/train/4_graphs.txt'
#     train_dataset = bAbIDataset(dataroot, 0, True)
#     loader = DataLoader(train_dataset, batch_size=2)
#     batch0 = next(iter(loader))
#     print(batch0.x, batch0.edge_index, batch0.batch, batch0.edge_attr, batch0.y)
