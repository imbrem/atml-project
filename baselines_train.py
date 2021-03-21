"""
RNN/LSTM baseline training for the bAbI dataset.

Adapted from Yujia Li,
https://github.com/yujiali/ggnn/blob/master/babi/babi_rnn_train.lua
"""


from data import load_rnn_data_from_file, find_max_in_list_of_tensors, split_set_tensor, split_set_input_output
from baselines.baseline_rnn import BaselineRNN
from torch import nn
import torch
import argparse
from pathlib import Path
import math

# def train(x, y):
#     optimizer.zero_grad()

#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)

#     loss = criterion(output, category_tensor)
#     loss.backward()

#     optimizer.step()

# ARGUMENT PARSING

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', default=1e-3,
                    type=float, help='learning rate')
# parser.add_argument('--momentum', default=0, type=float, help='momentum')
parser.add_argument('--batch_size', default=10,
                    type=int, help='minibatch size')
parser.add_argument('--n_epochs', default=0, type=int,
                    help='number of epochs to train, if this is set, maxiters will not be used')
parser.add_argument('--max_iters', default=1000, type=int,
                    help='maximum number of weight updates')
parser.add_argument('--print_every', default=10, type=int,
                    help='print training information after this number of weight updates')
parser.add_argument('--save_every', default=100, type=int,
                    help='save checkpoint after this number of weight updates')
parser.add_argument('--plot_every', default=10, type=int,
                    help='update training curves after this number of weight updates')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='type of optimization algorithm to use')
parser.add_argument('--embedding_size', default=50, type=int,
                    help='dimensionality of the embeddings')
parser.add_argument('--hidden_size', default=50, type=int,
                    help='dimensionality of the hidden layers')
parser.add_argument('--n_targets', default=1, type=int,
                    help='number of targets for each example, if > 1 the targets will be treated as a sequence')
parser.add_argument('--n_threads', default=1, type=int,
                    help='set the number of threads to use with this process')
parser.add_argument('--n_train', default=0, type=int,
                    help='number of training instances, 0 to use all available')
parser.add_argument('--n_val', default=50, type=int,
                    help='number of validation instances, this will not be used if datafile.val exists')
parser.add_argument('--output_dir', default='.',
                    type=str, help='output directory')
parser.add_argument('--data_file', default='', type=str,
                    help='should contain lists of edges and questions in standard format')
parser.add_argument('--model', default='rnn', type=str, help='rnn or lstm')
parser.add_argument('--seed', default=8, type=int, help='random seed')

# parser.add_argument('-gpuid', -1, 'ID of the GPU to use, not using any GPUs if <= 0')
args = parser.parse_args()


torch.set_num_threads(args.n_threads)

# TODO initialise RNN model here
rnn = None

if args.optimizer is not 'adam':
    raise argparse.ArgumentError('Unsupported optimizer')
optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)

# Set up checkpoint directory
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
print('Checkpoints will be saved to ', args.output_dir)
# torch.save(args, args.output_dir + '/params')

# PREPARE DATA

torch.manual_seed(args.seed)

# TODO fix load_rnn_data_from_file to only return tensors
x_train, t_train = load_rnn_data_from_file(args.data_file, args.n_targets)

# TODO get rid of this
# uniform length if x_train is a tensor, otherwise x_train is a list
uniform_length = torch.is_tensor(x_train)

if uniform_length:
    vocab_size = x_train.max()
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    output_size = t_train.max()
else:  # sequences of different lengths
    vocab_size = find_max_in_list_of_tensors(x_train)
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    output_size = find_max_in_list_of_tensors(t_train)


# split training data into train & val
# checking if validation data file exists
if Path(args.data_file + '.val').is_file():
    print('Validation file exists\nSplitting part of training data for validation.')
    if uniform_length:
        # TODO split_set_tensor
        x_train, t_train, x_val, t_val = split_set_tensor(
            x_train, t_train, args.n_train, args.n_val, True)
    else:
        # TODO split_set_input_output
        x_train, t_train, x_val, t_val = split_set_input_output(
            x_train, t_train, args.n_train, args.n_val, True)
else:
    # TODO isn't this behaving in the opposite way
    # if n_train is 0, automatically use all the training data available
    if args.n_train:
        if uniform_length:
            x_train, t_train = split_set_tensor(
                x_train, t_train, args.n_train, 0, True)
        else:
            x_train, t_train = split_set_input_output(
                x_train, t_train, args.n_train, 0, True)
    print('Loading validation data from {}.val'.format(args.data_file))
    x_val, t_val = load_rnn_data_from_file(
        args.data_file + '.val', args.n_targets)


if args.n_targets > 1:
    args.n_targets += 1  # add 1 to include the end symbol

print()

# TODO batch loading


def batch_loader(*args):
    pass

# TODO batch loading


def paired_data_loader(*args):
    pass


if uniform_length:
    # TODO batch loader
    train_data_loader = batch_loader(x_train, t_train, args.batch_size, True)
    print('Training set:\t{} sequences'.format(x_train.size()))
    print('Validation set:\t{} sequences'.format(x_val.size()))
    n_train = x_train.size(0)
else:
    # TODO paired data loader (?)
    train_data_loader = paired_data_loader(x_train, t_train, True)
    print('Training set:\t{} sequences'.format(x_train.size()))
    print('Validation set:\t{} sequences'.format(x_val))
    # TODO fix typing
    # because in this case x_train is assumed to be a tensor
    n_train = len(x_train)

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
params, grad_params = model.get_parameters()

print('Total number of parameters: {}\n'.format(params.nelement()))

# # TODO what does this do?
# def feval(x):
#     if x != params:
#         params.copy(x)
#     grad_param.zero()

#     if uniform_length:
#         x_batch, t_batch = train_data_loader.next()
#         if args.n_targets > 1:
#             t_batch = torch.reshape(t_batch, t_batch.size(0) * args.n_targets, t_batch.size(1) / args.n_targets)
#         y = model(x_batch, args.n_targets)
#         loss = criterion(y, t_batch)
#         loss.backward(y, t_batch)

#     else
#         loss=0
#         for i=1, opt.mb do
#             x_batch, t_batch=train_data_loader: next()
#             if opt.ntargets > 1 then
#                 t_batch=torch.reshape(t_batch, t_batch: size(1) * opt.ntargets, t_batch: size(2) / opt.ntargets)
#             end
#             y=net: forward(x_batch, opt.ntargets)
#             loss=loss + c: forward(y, t_batch)
#             net: backward(c: backward(y, t_batch))
#         end
#         loss=loss / opt.mb
#         grad_params: mul(1 / opt.mb)
#     end
#     grad_params: clamp(-5, 5)

#     return loss, grad_params
# end

# function eval_loss(model, x, t)
#     if uniform_length then
#         if opt.ntargets > 1 then
#             t=torch.reshape(t, t: size(1) * opt.ntargets, t: size(2) / opt.ntargets)
#         end
#         return c: forward(model: forward(x, opt.ntargets), t)
#     else
#         local loss=0
#         for i=1,  # x do
#             local tt=t[i]
#             if opt.ntargets > 1 then
#                 tt=torch.reshape(tt, tt: size(1) * opt.ntargets, tt: size(2) / opt.ntargets)
#             end
#             loss=loss + c: forward(model: forward(x[i], tt), tt)
#         end
#         return loss /  # x
#     end
# end

# function eval_err(model, x, t)
#     if uniform_length then
#         local pred=model: predict(x, opt.ntargets)
#         pred=pred: typeAs(t)
#         if opt.ntargets > 1 then
#             return pred: ne(t): type('torch.DoubleTensor'): sum(2): gt(0): type('torch.DoubleTensor'): mean()
#         else
#             return pred: ne(t): type('torch.DoubleTensor'): mean()
#         end
#     else
#         local err=0
#         for i=1,  # x do
#             local pred=model: predict(x[i], opt.ntargets)
#             if opt.ntargets > 1 then
#                 err=err + pred: typeAs(t[i]): ne(t[i]): type('torch.DoubleTensor'): sum(): gt(0): sum()
#             else
#                 err=err + pred: typeAs(t[i]): ne(t[i]): sum()
#             end
#         end
#         return err /  # x
#     end
# end

# train_records={}
# train_error_records={}
# val_records={}

# function train()
#     local loss=0
#     local batch_loss=0
#     local iter=0

#     local best_loss=math.huge
#     local best_params=params: clone()

#     local plot_iter=0

#     while iter < opt.maxiters do
#         local timer=torch.Timer()
#         batch_loss=0
#         for iter_before_print=1, opt.printafter do
#             _, loss=optfunc(feval, params, optim_config, optim_state)
#             loss=loss[1]
#             batch_loss=batch_loss + loss
#         end
#         iter=iter + opt.printafter
#         batch_loss=batch_loss / opt.printafter

#         plot_iter=plot_iter + 1

#         val_err=eval_err(net, x_val, t_val)
#         io.write(string.format('iter %d, grad_scale=%.8f, train_loss=%.6f, val_error_rate=%.6f, time=%.2f',
#                 iter, torch.abs(grad_params): max(), batch_loss, val_err, timer: time().real))

#         table.insert(train_records, {iter, batch_loss})
#         table.insert(val_records, {iter, val_err})

#         if val_err < best_loss then
#             best_loss=val_err
#             best_params: copy(params)
#             rnn.save_rnn_model(opt.outputdir .. '/model_best', best_params, opt.model, vocab_size, embed_size, hid_size, output_size)
#             print(color.green(' *'))
#         else
#             print('')
#         end

#         if iter % opt.saveafter == 0 then
#             rnn.save_rnn_model(opt.outputdir .. '/model_' .. iter, params, opt.model, vocab_size, embed_size, hid_size, output_size)
#         end

#         if plot_iter % opt.plotafter == 0 then
#             generate_plots()
#             collectgarbage()
#         end
#     end

#     generate_plots()

#     rnn.save_rnn_model(opt.outputdir .. '/model_end', params, opt.model, vocab_size, embed_size, hid_size, output_size)
# end

# function plot_learning_curve(records, fname, ylabel, xlabel)
#     xlabel=xlabel or '#iterations'
#     local rec=torch.Tensor(records)
#     gnuplot.pngfigure(opt.outputdir .. '/' .. fname .. '.png')
#     gnuplot.plot(rec: select(2, 1), rec: select(2, 2))
#     gnuplot.xlabel(xlabel)
#     gnuplot.ylabel(ylabel)
#     gnuplot.plotflush()
#     collectgarbage()
# end

# function generate_plots()
#     if not pcall(function() plot_learning_curve(train_records, 'train', 'training loss') end) then
#         print('[Warning] Failed to update training curve plot. Error ignored.')
#     end
#     if not pcall(function() plot_learning_curve(val_records, 'val', 'validation error rate') end) then
#         print('[Warning] Failed to update validation curve plot. Error ignored.')
#     end
#     # plot_learning_curve(train_records, 'train', 'training loss')
#     # plot_learning_curve(val_records, 'val', 'validation error rate')
#     if eval_train_err then
#         if not pcall(function() plot_learning_curve(train_error_records, 'train-err', 'training error rate') end) then
#             print(
#                 '[Warning] Failed to update training error curve plot. Error ignored.')
#         end
#         # plot_learning_curve(train_error_records,
#                                 'train-err', 'training error rate')
#     end
#     collectgarbage()
# end

# train()

# if __name__ == "__main__":
#     from torch_geometric.data import DataLoader

#     dataroot = 'babi_data/processed_1/train/4_graphs.txt'
#     train_dataset = bAbIDataset(dataroot, 0, True)
#     loader = DataLoader(train_dataset, batch_size=2)
#     batch0 = next(iter(loader))
#     print(batch0.x, batch0.edge_index, batch0.batch, batch0.edge_attr, batch0.y)
