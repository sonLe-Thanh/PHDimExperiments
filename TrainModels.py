import argparse

import torch
from torch import nn
from models.FC import FC
from models.AlexNet import AlexNet
from models.LeNet import LeNet
from PHDimPointCloud import estimatePersistentHomologyDimension, estimateMultiplePersistentHomologyDimension

import torch.optim as optim
from collections import deque

from utils import getData, accuracy
import numpy as np


def get_weights(model):
    with torch.no_grad():
        weights = []
        for param in model.parameters():
            # print(param.view(-1).detach().shape)
            weights.append(param.view(-1).detach().to(torch.device("cpu")))
        # print(weights)
        return torch.cat(weights)


def cycle_loader(data_loader):
    while True:
        for data in data_loader:
            yield data


def eval(loader, model, criterion, optimizer, args, data_type):
    # Change to eval behaviour
    model.eval()

    # Eval on both val and test set
    with torch.no_grad():
        total_size = 0
        total_loss = 0
        total_acc = 0
        # grads = []
        outputs = []

        # Same as in train, but not backward
        for input_batch, label_batch in loader:
            input_batch, label_batch = input_batch.to(args.device), label_batch.to(args.device)
            optimizer.zero_grad()
            output = model(input_batch)

            loss = criterion(output, label_batch)
            acc = accuracy(output, label_batch)
            batch_size = input_batch.size(0)

            total_size += int(batch_size)
            total_loss += float(loss.item()) * batch_size
            total_acc += float(acc.item()) * batch_size
            outputs.append(output)
    history = [total_loss / total_size, total_acc / total_size]
    print(f"{data_type}| Loss: {total_loss / total_size}| Accuracy: {total_acc/total_size}")
    return history, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", default=1000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist|cifar10|cifar100")
    parser.add_argument("--path", default="./data/", type=str)
    # parser.add_argument("--device", default="cpu", type=str)

    # Argument for model
    parser.add_argument("--model", default="FC", type=str, help="FC|AlexNet")

    # Arguments for optimzer
    parser.add_argument("--optimizer", default="SGD", type=str, help="SGD|Adam")
    # Learning rate for the optimization algorithm
    parser.add_argument("--learning_rate", default=0.1, type=float)

    # Width and depth for FC model
    parser.add_argument("--width", default=100, type=int)
    parser.add_argument("--depth", default=3, type=int)

    # Print some evaluation every
    parser.add_argument("--eval_every", default=100, type=int)

    # Where to save the data
    # parser.add_argument("--save_as", default="./results/PHDimReport/PHDimModel/FC.txt", type=str)

    args = parser.parse_args()

    # Use gpu if possible (not on Mac)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)

    train_loader, val_loader, test_loader, no_class = getData(args)

    if args.model == "FC":
        if args.dataset == "mnist":
            model = FC(input_width=args.width, depth=args.depth, no_class=no_class).to(args.device)
        elif args.dataset == "cifar10":
            model = FC(input_dim=3 * 32 * 32, input_width=args.width, depth=args.depth, no_class=no_class).to(
                args.device)
    elif args.model == "AlexNet":
        if args.dataset == "mnist":
            model = AlexNet(input_height=28, input_width=28, input_channels=1, no_class=no_class).to(args.device)
        else:
            model = AlexNet(no_class=no_class).to(args.device)

    print(model)
    print("....")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = getattr(optim, args.optimizer)(
        model.parameters(),
        lr=args.learning_rate
    )

    circ_train_loader = cycle_loader(train_loader)

    # Training history
    train_history = []

    # Eval val history
    eval_hist_test = []
    eval_hist_train = []

    # weights history
    weights_hist = deque([])

    is_stop = False

    for i, (input_batch, label_batch) in enumerate(circ_train_loader):

        # Eval for more information
        if i % args.eval_every == 0:
            # Eval here
            train_hist, train_outputs = eval(val_loader, model, criterion, optimizer, args, "Training set")
            test_hist, test_outputs = eval(test_loader, model, criterion, optimizer, args, "Test set")
            eval_hist_train.append([i, * train_hist])
            eval_hist_test.append([i, *test_hist])
            if int(train_hist[1]) == 1:
                is_stop = True

        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(input_batch)
        loss = criterion(output, label_batch)
        # print(loss.item())
        if torch.isnan(loss):
            is_stop = True

        # calculate gradients
        loss.backward()
        # Append: idx, val loss, acc
        train_history.append([i, loss.item(), accuracy(output, label_batch).item()])

        optimizer.step()

        if i > args.max_iter:
            is_stop = True

        # Get the weigths
        weights_hist.append(get_weights(model))

        # Only take at max 1000 according to the paper
        if len(weights_hist) > 1000:
            weights_hist.popleft()

        if is_stop:
            assert len(weights_hist) == 1000

            # Eval here
            train_hist, train_outputs = eval(val_loader, model, criterion, optimizer, args, "Training set")
            test_hist, test_outputs = eval(test_loader, model, criterion, optimizer, args, "Test set")

            eval_hist_train.append([i + 1, *train_hist])
            eval_hist_test.append([i + 1, *test_hist])

            weights_hist = torch.stack(tuple(weights_hist)).numpy()
            # Save weights for future experiments
            filename_weights = model.get_information_str() +"|"+args.dataset+ "|BatchSize:"+str(args.batch_size)+"|Opt:"+args.optimizer+"|LR:"+str(args.learning_rate)
            np.save("results/WeightsHistories/"+filename_weights, weights_hist)
            # _, S, _ = np.linalg.svd(weights_hist)
            # Calculate PH dim with weight hist
            max_dim = 1
            alpha = 1
            max_sampling_size = 1000
            PH_dim_model = estimateMultiplePersistentHomologyDimension(weights_hist.T, max_dim, alpha, max_sampling_size)

            test_acc = eval_hist_test[-1][2]
            train_acc = eval_hist_train[-1][2]

            # Save PH Dim
            full_path_PH_dim_model = "./results/PHDimReport/PHDimModel/"+args.model+".txt"
            with open(full_path_PH_dim_model, 'a') as file:
                if args.model == "FC":
                    for dim in range(max_dim+1):
                        file.write(f"{args.width}, {args.depth}, {args.learning_rate}, {args.dataset}, {args.batch_size}, {args.optimizer}, {train_acc}, {test_acc}, {dim}, {alpha}, {PH_dim_model[dim]}\n")
                elif args.model == "AlexNet":
                    for dim in range(max_dim+1):
                        file.write(f"{args.learning_rate}, {args.dataset}, {args.batch_size}, {args.optimizer}, {train_acc}, {test_acc}, {dim}, {alpha}, {PH_dim_model[dim]}\n")

            break
