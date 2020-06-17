# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 05: train_classifier.py
# Description:
#   - Approx. runtime on full data + server with GTX 1080 Ti: 6 min

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

from nets.classifier import FCClassifier

best_acc = 0

def train(args, dataset, model, optimizer, loss_fn, device, epoch):

    print("Epoch [%d/%d]" % (epoch+1, args.n_epoch), end=" | ")
    model.train()
    model = model.to(device=device)
    losses = []

    # Main training loop
    for x, y in dataset:

        # Setup for GPU
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # main training loop
        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Print average training loss across all batches
    mean_loss = np.mean(losses)
    print("Train loss: %0.5f" % mean_loss, end=" | ")

    # Save model
    torch.save(model.state_dict(), os.path.join("models", "fc_cls.pt"))

    return mean_loss


def test(args, dataset, model, optimizer, loss_fn, device, test=False):

    global best_acc
    global best_model

    model.eval()
    model = model.to(device=device)

    num_correct = 0
    num_samples = 0
    losses = []

    # Disable gradient updating
    with torch.no_grad():

        for x, y in dataset:

            # Setup for GPU
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Calculate predictions for each input hypercolumn
            y_score = model.forward(x)
            _, y_pred = torch.max(y_score, 1)

            # Calculate validation loss
            loss = loss_fn(y_score, y)
            losses.append(loss.item())

            # Accumulate accuracy counters
            num_correct += int(torch.sum(torch.eq(y_pred, y)))
            num_samples += len(y)

    acc = num_correct / num_samples
    if not test and acc > best_acc:
        best_model = model
        best_acc = acc

    # Print test results
    mean_loss = np.mean(losses)
    if test:
        print('Final test accuracy with best model:')
        print('%d/%d correct (%0.2f)' % (num_correct, num_samples, acc * 100))
        print('Test loss:', mean_loss)
    else:
        print('Val loss: %0.5f' % mean_loss, end=" | ")
        print('%d/%d correct (%0.2f)' % (num_correct, num_samples, acc * 100))

    return mean_loss


def plot_learning_curve(args, train_losses, val_losses):

    epochs = [i + 1 for i in range(args.n_epoch)]
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.legend()
    plt.savefig("learning_curve_fc.png")

    return None


def main():

    np.random.seed(seed=0)

    # Initialize batch job arguments
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--model_path', nargs='?', type=str, default='./models', help='Path to the saved models')
    parser.add_argument('--feature_path', nargs='?', type=str, default='/scratch/jonathantan/cv/features', help='Path to the saved hypercol features')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=40,    help='# of epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=64, help='Batch size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_hidden', nargs='?', type=int, default=1024, help='# of hidden units in MLP')
    parser.add_argument('--use_gpu', nargs='?', type=bool, default=True, help='Whether to use GPU if available')
    args = parser.parse_args()

    # Setup GPU settings
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Training using", device)

    # Select classifier and optimizer
    classifier = FCClassifier(device=device, n_hidden=args.n_hidden).float()
    optimizer = optim.Adam(classifier.parameters(), lr=args.l_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Transform numpy arrays to tensors, then wrap in TensorDataset class
    x_train = torch.Tensor(np.load(os.path.join(args.feature_path, "feats_x_train.npy")))
    y_train = torch.Tensor(np.load(os.path.join(args.feature_path, "feats_y_train.npy"))).long()
    dataset_train = data.TensorDataset(x_train, y_train)
    del x_train, y_train

    x_val = torch.Tensor(np.load(os.path.join(args.feature_path, "feats_x_val.npy")))
    y_val = torch.Tensor(np.load(os.path.join(args.feature_path, "feats_y_val.npy"))).long()
    dataset_val = data.TensorDataset(x_val, y_val)
    del x_val, y_val

    x_test = torch.Tensor(np.load(os.path.join(args.feature_path, "feats_x_test.npy")))
    y_test= torch.Tensor(np.load(os.path.join(args.feature_path, "feats_y_test.npy"))).long()
    dataset_test = data.TensorDataset(x_test, y_test)
    del x_test, y_test

    # Use inverse class weights for loss function - true distribution is 0.95-0.05
    WEIGHTS = torch.Tensor([0.4, 0.6]).to(device=device)
    loss_fn = nn.CrossEntropyLoss(weight=WEIGHTS)
    del WEIGHTS

    # Wrap data in dataloader classes, using 20% of data as validation set
    data_train = data.DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=True)
    data_val = data.DataLoader(dataset_val,
                          batch_size=args.batch_size,
                          num_workers=4,
                          shuffle=False)
    data_test = data.DataLoader(dataset_test,
                           batch_size=args.batch_size,
                           num_workers=4,
                           shuffle=False)
    del dataset_train, dataset_val, dataset_test

    # Training loop
    train_losses = []
    val_losses = []
    for epoch in range(args.n_epoch):
        train_loss = train(args, data_train, classifier, optimizer, loss_fn, device, epoch)
        val_loss = test(args, data_val, classifier, optimizer, loss_fn, device, test=False)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

    # Check overall accuracy on test set
    test(args, data_test, best_model, optimizer, loss_fn, device, test=True)

    # Export learning curve
    plot_learning_curve(args, train_losses, val_losses)

    # Save best model
    SAVE_PATH = os.path.join(args.model_path, "best_fc_dict.pt")
    torch.save(best_model.state_dict(), SAVE_PATH)


if __name__ == '__main__':
    main()
