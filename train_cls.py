# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 05: train_classifier.py
# Description:
#   -

import os
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as transforms

from nets.classifier import FCClassifier

best_acc = 0

def test(dataset, model, optimizer, loss_fn, device, test=False):

    global best_acc
    global best_model

    model.eval()
    model = model.to(device=device)

    num_correct = 0
    num_samples = 0
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

            # Accumulate counters
            num_correct += int(torch.sum(torch.eq(y_pred, y)))
            num_samples += len(y)

    acc = num_correct / num_samples
    if not test and acc > best_acc:
        best_model = model
        best_acc = acc

    if test:
        print('Final test accuracy with best model:')
        print(num_correct, '/', num_samples, 'correct (', round(acc * 100, 2), '%)')
        print('Test loss:', loss.item())
    else:
        print('\t', num_correct, '/', num_samples, 'correct (', round(acc * 100, 2), '%)')
        print('\t Validation loss:', loss.item())


def train(dataset, model, optimizer, loss_fn, device):

    model.train()
    model = model.to(device=device)

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
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(model, os.path.join("models", "fc_cls.pkl"))


def main():

    np.random.seed(seed=0)

    # Init path to directory where features are stored
    FEAT_PATH = os.path.join(os.sep, "scratch", "jonathantan", "features")

    # Setup GPU settings
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Select classifier and optimizer
    classifier = FCClassifier(device=device).float()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Transform numpy arrays to tensors, then wrap in TensorDataset class
    x_train = torch.Tensor(np.load(os.path.join(FEAT_PATH, "feats_x_train.npy")))
    y_train = torch.Tensor(np.load(os.path.join(FEAT_PATH, "feats_y_train.npy"))).long()
    dataset_train = data.TensorDataset(x_train, y_train)

    x_val = torch.Tensor(np.load(os.path.join(FEAT_PATH, "feats_x_test.npy")))
    y_val = torch.Tensor(np.load(os.path.join(FEAT_PATH, "feats_y_test.npy"))).long()
    dataset_val = data.TensorDataset(x_val, y_val)

    # Calculate inverse class weights for loss function
    _, class_counts = y_train.unique(return_counts = True)
    inv_counts = np.reciprocal(class_counts.float()).cuda()
    np.save(os.path.join(FEAT_PATH, "inv_counts.npy"), inv_counts.cpu())
    # inv_counts = torch.Tensor(np.load(os.path.join(FEAT_PATH, "inv_counts.npy")))
    loss_fn = nn.CrossEntropyLoss(weight=inv_counts)

    # Wrap data in dataloader classes, using 10% of data as validation set
    batch_size = 64
    num_val = int(0.1 * len(dataset_val))
    data_train = DataLoader(dataset_train,
                            batch_size=batch_size,
                            num_workers=4)
    data_val = DataLoader(dataset_val,
                          batch_size=batch_size,
                          num_workers=4,
                          sampler=sampler.SubsetRandomSampler(range(num_val)))
    data_test = DataLoader(dataset_val,
                           batch_size=batch_size,
                           num_workers=4,
                           sampler=sampler.SubsetRandomSampler(range(num_val, len(dataset_val))))

    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        print("Training epoch", epoch + 1, "of", num_epochs)
        train(data_train, classifier, optimizer, loss_fn, device)
        test(data_val, classifier, optimizer, loss_fn, device)
        scheduler.step()

    # Check overall accuracy on test set
    test(data_test, best_model, optimizer, loss_fn, device, test=True)
    torch.save(best_model, os.path.join("models", "best_fc_cls.pkl"))

if __name__ == '__main__':
    main()
