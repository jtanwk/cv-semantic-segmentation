# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 03: sampling.py
# Description:
#  - Generates hypercolumn features for each image and randomly samples a few
#       from each class to serve as training data.
#  - If GPU resources are available, can set "partial=False" in extract_samples
#       to use entire image rather than randomly selected pixels' hypercolumns.

import os
import torch
import numpy as np
from torch.utils import data

from nets.zoomout import Zoomout
from data.loader import xView2

def extract_samples(zoomout, dataset, partial=True):

    zoomout.eval()

    # Initialize empty lists and number of pixels per class to sample
    features = []
    features_labels = []
    num_samples = 3

    with torch.no_grad():

        for image_idx, (image, label) in enumerate(dataset):

            if image_idx % 1000 == 0:
                print('\t Parsing batch', image_idx + 1)

            # move to GPU if available
            image, label = image.float(), label.float()
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            # Extract hypercolumn features. zoom_feats is (C x H x W)
            zoom_feats = zoomout.forward(image.detach())

            if partial:
                # Randomly select a few pixels per unique value in labels
                for i in (0, 1):
                    num_idx = len(label[label == i])
                    if num_idx == 0:
                        continue
                    rand_idx = np.random.choice(range(num_idx), num_samples)
                    rand_px = np.argwhere(label.cpu() == i)[:, rand_idx]
                    hypercols = zoom_feats[:, rand_px[0], rand_px[1]]

                    # Store feature and labels in list to return
                    features.extend(hypercols.T.detach().clone().cpu().numpy())
                    features_labels.extend(np.repeat(i, num_samples))
            else:
                # Reshape (C x H x W) block into (H x W) array of C features
                C, H, W = zoom_feats.shape
                features = zoom_feats.reshape((C, H * W)).cpu().numpy().swapaxes(0, 1)
                features_labels = label.reshape((H * W)).cpu().numpy()

    return features, features_labels


def get_inverse_counts(dataset):
    '''
    Counts per-pixel occurence of labels and calculates inverse frequency.
    Useful for modifying loss function to avoid overweighting background class.
    '''

    labels = [x[1] for x in dataset]
    _, counts = torch.cat(labels).unique(return_counts = True)
    inv_counts = np.reciprocal(counts.float())

    return inv_counts


def main():

    np.random.seed(seed=0)

    # Init path to directory where features are stored
    FEAT_PATH = os.path.join(os.sep, "scratch", "jonathantan", "features")

    # Setup for GPU
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Initialize hypercolumn feature extractor
    zoomout = Zoomout().float().to(device=device)
    for param in zoomout.parameters():
        param.requires_grad = False

    # Extract separate samples for training, testing
    batch_size = 1
    dataset_train = xView2(split='train')
    train_loader = data.DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   num_workers=0,
                                   shuffle=True)
    dataset_test = xView2(split='test')
    test_loader = data.DataLoader(dataset_test,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=False)

    # Save features and labels
    print("Extracting zoomout features from training data")
    features_train, labels_train = extract_samples(zoomout, train_loader)
    np.save(os.path.join(FEAT_PATH, "feats_x_train.npy"), features_train)
    np.save(os.path.join(FEAT_PATH, "feats_y_train.npy"), labels_train)

    print("Extracting zoomout features from test data")
    features_test, labels_test = extract_samples(zoomout, test_loader)
    np.save(os.path.join(FEAT_PATH, "feats_x_test.npy"), features_test)
    np.save(os.path.join(FEAT_PATH, "feats_y_test.npy"), labels_test)

    # Save mean and sd of training features
    print("Extracting training means and standard deviations")
    means = np.mean(features_train, axis=0)
    stds = np.std(features_train, axis=0)
    np.save(os.path.join(FEAT_PATH, "mean_x_train.npy"), means)
    np.save(os.path.join(FEAT_PATH, "std_x_train.npy"), stds)

    # Save class counts for weighting loss function
    # inv_counts = get_inverse_counts(dataset_test)
    # np.save(os.path.join(FEAT_PATH, "inv_counts_all.npy"), inv_counts)


if __name__ == '__main__':
    main()
