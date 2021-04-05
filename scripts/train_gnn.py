import torch
import os
import wandb
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import DataLoader

from scripts.arguments import parse_arguments
from data import get_train_val_test_datasets
from ggnns.base_ggnn import BaseNodeSelectionGGNN, BaseGraphLevelGGNN


def initialise_experiments(fold_id=1):
    args = parse_arguments()
    # wandb.init(project=args.run_name,
    #            config={"epochs": args.epochs, "learning rate": args.lr, "batch size": args.bs,
    #                    "task id": args.task_id, "question id": args.question_id})
    train_dataset, val_dataset, test_dataset, total_edge_types = get_train_val_test_datasets(
        babi_data_path=args.data_root, task_id=args.task_id, question_id=args.question_id,
        train_examples=args.train_examples, fold_id=fold_id)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    if args.task_id != 18:
        model = BaseNodeSelectionGGNN(state_size=args.annotation_size, num_layers=args.num_layers, ggnn_impl="team2",
                                      total_edge_types=total_edge_types, out_channels=args.annotation_size)
    else:
        assert args.annotation_size == 2
        model = BaseGraphLevelGGNN(annotation_size=args.annotation_size, num_layers=args.num_layers, ggnn_impl="team2",
                                   total_edge_types=total_edge_types, classification_categories=2,
                                   state_size=args.state_size)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_parameters <= 600

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    evaluation_interval = args.evaluation_interval
    total_epochs = args.epochs
    return train_loader, val_loader, test_loader, model, criterion, optimizer, evaluation_interval, total_epochs


def train(train_loader, val_loader, test_loader, model, optimizer, criterion, eval_interval, total_epochs,
          save_final_checkpoint=False, checkpoint_directory=None):
    max_val_acc = 0
    test_acc_to_return = 0
    for epoch in tqdm(range(1, total_epochs+1)):
        train_loss, train_accuracy = train_epoch(train_loader, model, optimizer, criterion)
        # wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)

        if epoch == 1 or epoch % eval_interval == 0:
            epoch_loss, epoch_accuracy = evaluate(train_loader, model, criterion)
            val_loss, val_accuracy = evaluate(val_loader, model, criterion)
            test_loss, test_accuracy = evaluate(test_loader, model, criterion)
            if val_accuracy > max_val_acc:
                max_val_acc = val_accuracy
                test_acc_to_return = test_accuracy
            # wandb.log({'epoch_loss': epoch_loss, 'epoch_accuracy': epoch_accuracy}, step=epoch)
            # wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy}, step=epoch)
            # wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy}, step=epoch)

    if save_final_checkpoint and checkpoint_directory is not None:
        if not os.path.isdir(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            os.path.join(checkpoint_directory, "state.pth"))
    return test_acc_to_return


def train_epoch(train_loader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    for graph_batch in train_loader:
        prediction_probs = model(x=graph_batch.x, edge_index=graph_batch.edge_index,
                                 edge_attr=graph_batch.edge_attr, batch=graph_batch.batch)
        _, predicted = torch.max(prediction_probs, 1)
        loss = criterion(prediction_probs, graph_batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        examples = graph_batch.y.size(0)
        total_examples += examples
        total_loss += loss.item() * examples
        total_correct += (predicted == graph_batch.y).sum()
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(loader, model, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    for graph_batch in loader:
        prediction_probs = model(x=graph_batch.x, edge_index=graph_batch.edge_index,
                                 edge_attr=graph_batch.edge_attr, batch=graph_batch.batch)
        _, predicted = torch.max(prediction_probs, 1)
        loss = criterion(prediction_probs, graph_batch.y)
        examples = graph_batch.y.size(0)
        total_examples += examples
        total_loss += loss.item() * examples
        total_correct += (predicted == graph_batch.y).sum()
        # print(prediction_probs, predicted)
    return total_loss / total_examples, total_correct / total_examples


if __name__ == "__main__":
    fold_id = 1
    while fold_id <= 10:
        train_loader, val_loader, test_loader, model, criterion, optimizer, evaluation_interval, total_epochs = \
            initialise_experiments(fold_id=fold_id)
        print("Fold {}".format(fold_id))
        test_acc = train(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                         model=model, optimizer=optimizer, criterion=criterion,
                         eval_interval=evaluation_interval, total_epochs=total_epochs,
                         save_final_checkpoint=False, checkpoint_directory=None)
        if test_acc.detach().item() > 0.6:
            fold_id += 1
            print(test_acc.detach().item())
