import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Experimental settings.')
    parser.add_argument('-lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('-bs', type=int, default=20, help="batch size")
    parser.add_argument('--num-layers', '-nl', type=int, default=5, help="number of layers")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="total epochs to run")
    parser.add_argument('--evaluation-interval', '-ei', type=int, default=5, help="evluation interval")
    parser.add_argument('--run-name', '-rn', type=str, default="atml-project", help="name of the run")
    parser.add_argument('--data-root', '-dr', type=str, default="../babi_data", help="place to store the data")

    parser.add_argument('--question-id', '-qi', type=int, default=0, help="type of the question")
    parser.add_argument('--task-id', '-ti', type=int, default=4, help="type of the task")
    parser.add_argument('--annotation-size', '-as', type=int, default=6, help="size of annotations")
    parser.add_argument('--edge-attr-size', '-eas', type=int, default=4, help="size of edge attributes")
    parser.add_argument('--train-examples', '-te', type=int, default=50, help="total training examples to use")
    return parser.parse_args()
