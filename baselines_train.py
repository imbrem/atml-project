"""
RNN/LSTM baseline training for the bAbI dataset.

Adapted from Yujia Li,
https://github.com/yujiali/ggnn/blob/master/babi/babi_rnn_train.lua
"""


from baselines_data import load_rnn_data_from_file, find_max_in_list_of_tensors, split_set_tensor, split_set_input_output
from baselines.baseline_rnn import BaselineRNN
from torch import nn
import torch
import argparse
from pathlib import Path
import math

# ARGUMENT PARSING

parser = argparse.ArgumentParser()

# autopep8: off
# Model parameters
parser.add_argument('--model', default='rnn', type=str, help='rnn or lstm')
parser.add_argument('--embedding_size', default=50, type=int, help='dimensionality of the embeddings')
parser.add_argument('--hidden_size', default=50, type=int, help='dimensionality of the hidden layers')
parser.add_argument('--n_targets', default=1, type=int, help='number of targets for each example, if > 1 the targets will be treated as a sequence')

# Data parameters
parser.add_argument('--data_file', default='', type=str)
parser.add_argument('--n_train', default=0, type=int, help='number of training instances, 0 to use all available')
parser.add_argument('--n_val', default=50, type=int, help='number of validation instances (will not be used if datafile.val exists)')

# Training parameters
parser.add_argument('--learning_rate', default=1e-3,type=float, help='learning rate')
parser.add_argument('--optimizer', default='adam', type=str, help='type of optimization algorithm to use')
parser.add_argument('--batch_size', default=10,type=int, help='minibatch size')
parser.add_argument('--n_epochs', default=0, type=int, help='number of epochs to train, (overrides maxiters)')
parser.add_argument('--max_iters', default=1000, type=int, help='maximum number of weight updates')
parser.add_argument('--n_threads', default=1, type=int, help='set the number of threads to use with this process')

# Checkpoint and logging parameters
parser.add_argument('--output_dir', default='.', type=str, help='output directory')
parser.add_argument('--print_every', default=10, type=int, help='frequency of training information logs')
parser.add_argument('--save_every', default=100, type=int, help='frequency of checkpoints')
parser.add_argument('--plot_every', default=10, type=int, help='frequency of training curve updates')

parser.add_argument('--seed', default=8, type=int, help='random seed')
# autopep8: on

args = parser.parse_args()

# Set up checkpoint directory
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
print('Checkpoints will be saved to ', args.output_dir)


torch.set_num_threads(args.n_threads)

# TODO initialise RNN model here
rnn = None

if args.optimizer is not 'adam':
    raise argparse.ArgumentError('Unsupported optimizer')
optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)


# PREPARE DATA

torch.manual_seed(args.seed)

seq_train, target_train = load_rnn_data_from_file(
    args.data_file, args.n_targets)

vocab_size = seq_train.max()
embedding_size = args.embedding_size
hidden_size = args.hidden_size
output_size = target_train.max()


# split training data into train & val
# checking if validation data file exists
if Path(args.data_file + '.val').is_file():
    print('Validation file exists\nSplitting part of training data for validation.')
    # TODO split_set_tensor
    seq_train, target_train, seq_val, target_val = split_set_tensor(
        seq_train, target_train, args.n_train, args.n_val, True)
else:
    # TODO isn't this behaving in the opposite way
    # if n_train is 0, automatically use all the training data available
    if args.n_train:
        seq_train, target_train = split_set_tensor(
            seq_train, target_train, args.n_train, 0, True)

    print('Loading validation data from {}.val'.format(args.data_file))
    seq_val, target_val = load_rnn_data_from_file(
        args.data_file + '.val', args.n_targets)


if args.n_targets > 1:
    args.n_targets += 1  # add 1 to include the end symbol

print()

# TODO batch loading


def batch_loader(*args):
    pass


train_data_loader = batch_loader(
    seq_train, target_train, args.batch_size, True)
print('Training set:\t{} sequences'.format(seq_train.size()))
print('Validation set:\t{} sequences'.format(seq_val.size()))
n_train = seq_train.size(0)

# PREPARE MODEL

# TODO separate printing

print('Number of output classes: {}\n'.format(output_size))

if args.n_epochs > 0:
    args.max_iters = args.n_epochs * math.ceil(n_train * 1. / args.batch_size)
print('Total number of weight updates: {}\n'.format(args.max_iters))

# TODO fix model initialisation
if args.model == 'rnn':
    model = rnn.RNN(vocab_size, embedding_size, hidden_size, output_size)
# elif args.model == 'lstm':
#     model =rnn.LSTM(vocab_size, embedding_size, hidden_size, output_size)
else:
    argparse.ArgumentError('Unknown model type: {}'.format(args.model))

# TODO get rid of criterion
criterion = nn.CrossEntropyCriterion()

model.print_model()
print('Loss: {}\n'.format(criterion))

# TODO get parameters
# TODO get rid of them
params, grad_params = model.get_parameters()

print('Total number of parameters: {}\n'.format(params.nelement()))

# EVALUATION CODE


def forward_evaluation(x):
    # TODO what does this do?
    if x != params:
        params.copy(x)
    optimizer.zero_grad()

    x_batch, t_batch = train_data_loader.next()
    if args.n_targets > 1:
        t_batch = torch.reshape(t_batch, t_batch.size(
            0) * args.n_targets, t_batch.size(1) / args.n_targets)
    y = model(x_batch, args.n_targets)
    loss = criterion(y, t_batch)
    loss.backward()

    # TODO clamping
    # grad_params: clamp(-5, 5)
    optimizer.step()

    return loss, grad_params


def eval_loss(model, x, t):
    if args.n_targets > 1:
        t = torch.reshape(t, t.size(0) * args.n_targets,
                          t.size(1) / args.n_targets)
    return criterion(model(x, args.n_targets), t)


def eval_error(model, x, t):
    pred = model.predict(x, args.n_targets)
    # TODO why is this type conversion...
    # pred = pred.typeAs(t)
    if args.n_targets > 1:
        # TODO wth
        return pred.ne(t).type('torch.DoubleTensor').sum(1).gt(0).type('torch.DoubleTensor').mean()
    else:
        # TODO fix
        return pred.ne(t).type('torch.DoubleTensor').mean()


train_records = []
train_error_records = []
val_records = []

# TRAINING


def train():
    total_loss, batch_loss, iter = 0, 0, 0

    best_loss = math.inf
    best_params = params.clone()

    plot_iter = 0

    while iter < args.max_iters:
        # TODO timer
        timer = torch.Timer()
        batch_loss = 0
        for _ in range(args.print_every):
            # TODO compute loss
            # _, loss = optfunc(feval, params, optim_config, optim_state)
            loss = None
            batch_loss = batch_loss + loss
        iter += args.print_every
        batch_loss /= args.print_every

        plot_iter += 1

        val_err = eval_error(model, seq_val, target_val)
        print('iter {%d}, grad_scale={%.8f}, train_loss={%.6f}, val_error_rate={%.6f}, time={%.2f}'.format(
            iter, torch.abs(grad_params).max(), batch_loss, val_err, timer.time().real))

        train_records.append([iter, batch_loss])
        val_records.append([iter, val_err])

        # TODO off-the-shelf early stopping
        if val_err < best_loss:
            best_loss = val_err
            # TODO save best parameters
            best_params = params.clone()
            # TODO torch.save() model
            # model.save_rnn_model(opt.outputdir .. '/model_best', best_params, opt.model, vocab_size, embed_size, hid_size, output_size)
            # print(color.green(' *'))
        else:
            print('')

        if iter % args.save_every == 0:
            # TODO torch.save() the model
            # rnn.save_rnn_model(opt.outputdir .. '/model_' .. iter, params, opt.model, vocab_size, embed_size, hid_size, output_size)
            pass

        if plot_iter % args.plot_every == 0:
            # TODO fill
            pass
            # generate_plots()
            # collectgarbage()

    # generate_plots()
    # TODO torch.save() the final model
    # rnn.save_rnn_model(opt.outputdir .. '/model_end', params, opt.model, vocab_size, embed_size, hid_size, output_size)

# TODO plotting

# TODO if __name__ == "__main__":
# train()

# if __name__ == "__main__":
#     from torch_geometric.data import DataLoader

#     dataroot = 'babi_data/processed_1/train/4_graphs.txt'
#     train_dataset = bAbIDataset(dataroot, 0, True)
#     loader = DataLoader(train_dataset, batch_size=2)
#     batch0 = next(iter(loader))
#     print(batch0.x, batch0.edge_index, batch0.batch, batch0.edge_attr, batch0.y)


# def train(x, y):
#     optimizer.zero_grad()

#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)

#     loss = criterion(output, category_tensor)
#     loss.backward()

#     optimizer.step()
