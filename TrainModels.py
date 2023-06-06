import argparse

import torch
from torch import nn
from models.FC import FC
from models.AlexNet import AlexNet
from PHDimPointCloud import computeTopologyDescriptors

import torch.optim as optim
from collections import deque, OrderedDict

from utils import getData

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


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
        total_loss = 0
        correct = 0
        # grads = []
        outputs = []

        # Same as in train, but not backward
        for input_batch, label_batch in loader:
            input_batch, label_batch = input_batch.to(args.device), label_batch.to(args.device)
            optimizer.zero_grad()
            output = model(input_batch)

            loss = criterion(output, label_batch)

            # acc = accuracy(output, label_batch)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label_batch.data.view_as(pred)).sum().item()

            total_loss += float(loss.detach().item()) * len(label_batch)
            outputs.append(output)

    history = [total_loss / len(loader.dataset), correct / len(loader.dataset)]
    print(
        f"Test on all {data_type}| Loss: {total_loss / len(loader.dataset)}| Accuracy: {correct / len(loader.dataset)}")
    return history, outputs

# Wrapper function for not repeating twice
def evalTopoDescriptors(mid_outputs, output, weights_hist, path, alpha, counter, train_history, test_history):
    # Get the representations after each intermediate layers
    for layer in mid_outputs:
        output_layer = mid_outputs[layer].detach().numpy()
        e_0, e_1, entropy_0, entropy_1, entropy_total, ph_dim_info = computeTopologyDescriptors(
            output_layer, 1, alpha)
        # Save
        with open(path, 'a') as file:
            if args.model == "FC":
                file.write(
                    f"{counter}, {train_history[0]}, {train_history[1]}, {test_history[0]}, {test_history[1]}, {layer}, {e_0}, {e_1}, {entropy_0}, {entropy_1}, {entropy_total}, {ph_dim_info}\n")
    # Descriptors for output
    output_cal = output.detach().numpy()
    e_0_o, e_1_o, entropy_0_o, entropy_1_o, entropy_total_o, ph_dim_info_o = computeTopologyDescriptors(
        output_cal, 1, alpha)

    # Descriptors for weights
    weights_hist_cal = torch.stack(tuple(weights_hist)).numpy()

    e_0_w, e_1_w, entropy_0_w, entropy_1_w, entropy_total_w, ph_dim_info_w = computeTopologyDescriptors(
        weights_hist_cal.T, 1, alpha)
    # Save
    with open(path, 'a') as file:
        if args.model == "FC":
            file.write(
                f"{counter}, {train_history[0]}, {train_history[1]}, {test_history[0]}, {test_history[1]}, output, {e_0_o}, {e_1_o}, {entropy_0_o}, {entropy_1_o}, {entropy_total_o}, {ph_dim_info_o}\n"
                f"{counter}, {train_history[0]}, {train_history[1]}, {test_history[0]}, {test_history[1]}, weights, {e_0_w}, {e_1_w}, {entropy_0_w}, {entropy_1_o}, {entropy_total_w}, {ph_dim_info_w}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", default=1000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist|cifar10|cifar100|test_data1")
    parser.add_argument("--path", default="./data/", type=str)
    parser.add_argument("--device", default="cpu", type=str, help="cpu|gpu")

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

    print(model)
    print("....")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = getattr(optim, args.optimizer)(
        model.parameters(),
        lr=args.learning_rate
    )

    mid_getter = MidGetter(model, return_layers, keep_output=True)

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
    is_switch_eval_scheme = False
    path_descp = "./results/TopologicalDescriptors/FC/D5W100_MNIST/descrip.txt"
    alpha = 1
    for epochs in range(1, args.max_iter + 1):
        print("Epoch % --------------- %")
        for i, (input_batch, label_batch) in enumerate(train_loader):
            input_batch = input_batch.to(args.device)
            label_batch = label_batch.to(args.device)
            model.train()
            optimizer.zero_grad()

            # Forward pass
            # output = model(input_batch)
            mid_outputs, output = mid_getter(input_batch)

            # print(mid_outputs)
            # print(output)
            loss = criterion(output, label_batch)
            # print(loss.item())
            if torch.isnan(loss):
                is_stop = True

            # calculate gradients
            loss.backward()
            # Append: idx, val loss, acc
            optimizer.step()

            # Get the weigths
            weights_hist.append(get_weights(model))

            # Only take at max 1000 according to the paper
            # if len(weights_hist) > 1000:
            #     weights_hist.popleft()

            # Take 2000 for stability
            if len(weights_hist) > 2000:
                weights_hist.popleft()

            counter = i * args.batch_size + (epochs - 1) * len(train_loader.dataset)
            if counter % args.eval_every == 0 and not is_switch_eval_scheme:
                # Evaluate every interval
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epochs, i * len(input_batch), len(train_loader.dataset),
                            100. * i / len(train_loader), loss.item()))
                train_hist, train_outputs = eval(val_loader, model, criterion, optimizer, args, "Training set")
                test_hist, test_outputs = eval(test_loader, model, criterion, optimizer, args, "Test set")
                train_counter.append(counter)
                # Only append the loss
                train_history.append(train_hist[1])

                if train_hist[1] >= 0.6 and counter >= 2000:
                    is_switch_eval_scheme = True
                    # Evaluate the topological descriptors once accuracy reaches 0.6 and train_counter > 2000 (for weight
                    # space)
                    evalTopoDescriptors(mid_outputs, output, weights_hist, path_descp, alpha, counter, train_hist, test_hist)
                # stop if achieve acc 98% on training set
                if train_hist[1] > 0.98:
                    is_stop = True
            elif is_switch_eval_scheme and counter % args.eval_every_acc:
                train_hist, train_outputs = eval(val_loader, model, criterion, optimizer, args, "Training set")
                test_hist, test_outputs = eval(test_loader, model, criterion, optimizer, args, "Test set")
                train_counter.append(counter)
                evalTopoDescriptors(mid_outputs, output, weights_hist, path_descp, alpha, counter, train_hist, test_hist)
                if train_hist[1] > 0.98:
                    is_stop = True

            if is_stop:
                break
        # Evaluate again at the end of each epoch
        print(f"End of epoch {epochs}")
        train_hist, train_outputs = eval(val_loader, model, criterion, optimizer, args, "Training set")
        test_hist, test_outputs = eval(test_loader, model, criterion, optimizer, args, "Test set")
        # stop if achieve acc 99% on training set
        if train_hist[1] > 0.98:
            break
