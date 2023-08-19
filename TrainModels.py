import argparse

import torch
from torch import nn
from models.FC import FC
from models.ResNet import resnet18, resnet34

import torch.optim as optim
from collections import deque, OrderedDict

from utils import getData
import numpy as np
from Attack import *


def get_weights(model):
    with torch.no_grad():
        weights = []
        for param in model.parameters():
            # print(param.view(-1).detach().shape)
            weights.append(param.view(-1).detach().to(torch.device("cpu")))
        # print(weights)
        return torch.cat(weights)


def accuracy(output, labels):
    _, pred = output.max(1)
    correct_labels = pred.eq(labels)
    return correct_labels.sum().float() / labels.size(0)


def eval(loader, model, criterion, optimizer, args, data_type):
    # Change to eval behaviour
    model.eval()

    # Eval on both val and test set
    with torch.no_grad():
        total_point = 0
        total_loss = 0
        correct = 0
        outputs = []

        # Same as in train, but not backward
        for input_batch, label_batch in loader:
            input_batch, label_batch = input_batch.to(args.device), label_batch.to(args.device)
            optimizer.zero_grad()
            output = model(input_batch)

            loss = criterion(output, label_batch)

            # acc = accuracy(output, label_batch)

            pred = torch.max(output.data, 1)[1]
            correct += (pred == label_batch).sum().item()
            total_point += label_batch.size(0)

            total_loss += float(loss.detach().item()) * len(label_batch)
            outputs.append(output)

    history = [total_loss / total_point, correct / total_point]
    print(f"Test on all {data_type}| Loss: {total_loss / total_point}| Accuracy: {correct / total_point}")
    return history, outputs


# Only train model now
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist|cifar10|cifar100|test_data1")
    parser.add_argument("--path", default="./data/", type=str)
    parser.add_argument("--device", default="cpu", type=str, help="cpu|gpu")

    # Argument for model
    parser.add_argument("--model", default="FC", type=str, help="FC|AlexNet|ResNet18")
    parser.add_argument("--adversarial", default=0, type=int, help="0:normal training|1:adversarial training")

    # Arguments for optimzer
    parser.add_argument("--optimizer", default="SGD", type=str, help="SGD|Adam")
    # Learning rate for the optimization algorithm
    parser.add_argument("--learning_rate", default=0.1, type=float)

    # Width and depth for FC model
    parser.add_argument("--width", default=100, type=int)
    parser.add_argument("--depth", default=3, type=int)

    # Print some evaluation every
    parser.add_argument("--eval_every", default=2000, type=int)
    # Print some evaluation every after training reached acc > 90%
    parser.add_argument("--eval_every_acc", default=100, type=int)

    # Where to save the data
    # parser.add_argument("--save_as", default="./results/PHDimReport/PHDimModel/FC.txt", type=str)

    args = parser.parse_args()

    # Use gpu if possible (not on Mac)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)

    train_loader, val_loader, test_loader, no_class = getData(args)
    return_layers = OrderedDict()
    if args.model == "FC":
        if args.dataset == "mnist":
            model = FC(input_width=args.width, depth=args.depth, no_class=no_class).to(args.device)
        elif args.dataset == "cifar10":
            model = FC(input_dim=3 * 32 * 32, input_width=args.width, depth=args.depth, no_class=no_class).to(
                args.device)
        elif args.dataset == "test_data1":
            model = FC(input_dim=3, input_width=args.width, depth=args.depth, no_class=no_class).to(
                args.device)
        for i in range(len(model.features)):
            if i % 2 == 1:
                return_layers['features.' + str(i)] = 'layer' + str(int(i / 2) + 1)
    elif args.model == "AlexNet":
        if args.dataset == "mnist":
            model = AlexNet(input_height=28, input_width=28, input_channels=1, no_class=no_class).to(args.device)
        else:
            model = AlexNet(no_class=no_class).to(args.device)
    elif args.model == "ResNet18":
        if args.dataset == "mnist":
            model = resnet18(no_class, True)
        else:
            model = resnet18(no_class, False)
    elif args.model == "ResNet34":
        if args.dataset == "mnist":
            model = resnet34(no_class, True)
        else:
            model = resnet34(no_class, False)

    print(model)
    print("....")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = getattr(optim, args.optimizer)(
        model.parameters(),
        lr=args.learning_rate
    )


    # Training history
    train_history = []

    # Eval val history
    eval_hist_test = []
    eval_hist_train = []

    # weights history
    weights_hist = deque([])

    # Train counter
    train_counter = []

    is_stop = False
    path_descp = "results/TrainedModels/AlexNet_MNIST/"
    for epochs in range(1, args.max_epoch + 1):
        print("Epoch % --------------- %")
        for i, (input_batch, label_batch) in enumerate(train_loader):
            input_batch = input_batch.to(args.device)
            label_batch = label_batch.to(args.device)
            model.train()


            if args.adversarial == 1:
                print("Create adversarial examples")
                # Perform adversarial training
                epsilon = 0.1
                delta = fgsmAttack(model, input_batch, label_batch, epsilon, criterion)

                # Restore the data to its original scale
                data_denorm = denormalize(input_batch, device=args.device)
                perturbed_data = data_denorm + delta
                # Clamp to (0,1)
                perturbed_data = torch.clamp(perturbed_data, 0, 1)
                # Reapply normalization
                perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

                output_perturbed = model(perturbed_data_normalized)
                loss = criterion(output_perturbed, label_batch)
            else:
                # Forward pass
                output = model(input_batch)
                loss = criterion(output, label_batch)
                # print(loss.item())
            if torch.isnan(loss):
                is_stop = True

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get the weigths
            weights_hist.append(get_weights(model))

            # Only take at max 1000 according to the paper
            # if len(weights_hist) > 1000:
            #     weights_hist.popleft()

            # Take 2000 for stability
            if len(weights_hist) > 2000:
                weights_hist.popleft()

            counter = i + (epochs - 1) * int(len(train_loader.dataset) / len(input_batch))
            # print(counter)
            if counter % args.eval_every == 0:
                # Evaluate every interval
                print('Train Epoch: {} Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epochs, counter, i * len(input_batch), len(train_loader.dataset),
                            100. * i / len(train_loader), loss.item()))
                train_hist, train_outputs = eval(val_loader, model, criterion, optimizer, args, "Training set")
                test_hist, test_outputs = eval(test_loader, model, criterion, optimizer, args, "Test set")
                train_counter.append(counter)
                # Only append the loss
                train_history.append(train_hist[1])


                # stop if achieve acc 96% on training set
                if train_hist[1] >= 0.98:
                    is_stop = True


            if is_stop:
                break
        # Evaluate again at the end of each epoch
        print(f"End of epoch {epochs}")
        train_hist, train_outputs = eval(val_loader, model, criterion, optimizer, args, "Training set")
        test_hist, test_outputs = eval(test_loader, model, criterion, optimizer, args, "Test set")
        # stop if achieve acc 99% on training set
        if train_hist[1] >= 0.98:
            break
    # Save models, save weights
    torch.save(model.state_dict(), path_descp+"AlexNetRobust1.pth")
    np.save(path_descp+"AlexNetRobust_Weights1.npy", torch.stack(tuple(weights_hist)).numpy())

# python TrainModels.py --max_epoch 100 --batch_size 100 --dataset cifar10_missing --model AlexNet_CIFAR10 --eval_every 2000 --eval_every_acc 100