# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 03: classifier.py
# Description:
#   -

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FCClassifier(nn.Module):

    def __init__(self, device, n_hidden, n_classes=2):
        super(FCClassifier, self).__init__()

        # Compute mean and sd for manual normalization later
        FEAT_PATH = os.path.join(os.sep, "scratch", "jonathantan", "cv", "features")
        self.mean = torch.Tensor(np.load(os.path.join(FEAT_PATH, "mean_x_train.npy")))
        self.std = torch.Tensor(np.load(os.path.join(FEAT_PATH, "std_x_train.npy")))
        input_dim = self.mean.shape[0]
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        # Setup 3 linear layers
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)

        # Dropout layer with default dropout rate of 0.5
        self.dropout = nn.Dropout()

    def forward(self, x):

        # normalization
        x = (x - self.mean) / (self.std)

        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x, inplace=False)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class DenseClassifier(nn.Module):

    def __init__(self, fc_model, device, n_hidden, n_classes=2):
        super(DenseClassifier, self).__init__()

        # Load stored training mean and std
        FEAT_PATH = os.path.join(os.sep, "scratch", "jonathantan", "cv", "features")
        self.mean = torch.Tensor(np.load(os.path.join(FEAT_PATH, "mean_x_train.npy")))
        self.std = torch.Tensor(np.load(os.path.join(FEAT_PATH, "std_x_train.npy")))
        input_dim = self.mean.shape[0]
        self.mean = torch.Tensor(np.expand_dims(np.expand_dims(self.mean, -1), -1)).to(device)
        self.std = torch.Tensor(np.expand_dims(np.expand_dims(self.std, -1), -1)).to(device)

        # Setup 3 convolutional layers
        self.conv1 = nn.Conv2d(input_dim, n_hidden, (1, 1), stride=2, padding=0)
        self.conv2 = nn.Conv2d(n_hidden, n_hidden, (1, 1), stride=2, padding=0)
        self.conv3 = nn.Conv2d(n_hidden, n_classes, (1, 1), stride=1, padding=0)

        # Load pretrained FC classifier weights and reshape
        weights = fc_model.state_dict()
        self.conv1.weight = nn.Parameter(weights['fc1.weight'].view(n_hidden, input_dim, 1, 1))
        self.conv1.bias = nn.Parameter(weights['fc1.bias'])
        self.conv2.weight = nn.Parameter(weights['fc2.weight'].view(n_hidden, n_hidden, 1, 1))
        self.conv2.bias = nn.Parameter(weights['fc2.bias'])
        self.conv3.weight = nn.Parameter(weights['fc3.weight'].view(n_classes, n_hidden, 1, 1))
        self.conv3.bias = nn.Parameter(weights['fc3.bias'])
        del weights


    def forward(self, x):

        # normalization
        x = (x - self.mean) / (self.std)

        # 1x1 convolutional layers
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)

        # Upsample to original H and W using bilinear interpolation
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)

        return x
