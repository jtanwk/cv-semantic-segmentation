# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 04: sampling.py
# Description:
#  - Generates hypercolumn features for each image and randomly samples a few
#       from each class to serve as training data.
#  - If GPU resources are available, can set "partial=False" in extract_samples
#       to use entire image rather than randomly selected pixels' hypercolumns.
# - Approx. runtime on full data + server with GTX 1080 Ti: 45 minutes

import os
import argparse
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
                    features.extend(hypercols.T.detach().cpu().numpy())
                    features_labels.extend(np.repeat(i, num_samples))
            else:
                # Reshape (C x H x W) block into (H x W) array of C features
                C, H, W = zoom_feats.shape
                features = zoom_feats.reshape((C, H * W)).detach().cpu().numpy().swapaxes(0, 1)
                features_labels = label.reshape((H * W)).detach().cpu().numpy()

    return features, features_labels


def save_features(args, zoomout, split):

    # Load data and wrap in dataloader
    dataset = xView2(split=split)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  num_workers=0)
    del dataset

    # Extract hypercolumns and save
    print("Extracting hypercolumn features from %s data" % split)
    features, labels = extract_samples(zoomout, data_loader)
    np.save(os.path.join(args.feature_path, "feats_x_" + split + ".npy"), features)
    np.save(os.path.join(args.feature_path, "feats_y_" + split + ".npy"), labels)
    del data_loader

    # If training data, also extract mean and standard deviation for all features
    if split == "train":
        print("Extracting means and standard deviations from train data")
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        np.save(os.path.join(args.feature_path, "mean_x_train.npy"), means)
        np.save(os.path.join(args.feature_path, "std_x_train.npy"), stds)
        del means, stds

    del features, labels

    return None


def main():

    np.random.seed(seed=0)

    # Initialize batch job arguments
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--feature_path', nargs='?', type=str, default='/scratch/jonathantan/cv/features', help='Directory to save hypercols to')
    parser.add_argument('--out_path', nargs='?', type=str, default='/scratch/jonathantan/cv/results', help='Directory to save results to')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,    help='# of epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,  help='Batch size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_gpu', nargs='?', type=bool, default=True, help='Whether to use GPU if available')
    args = parser.parse_args()

    # Setup for GPU
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Initialize hypercolumn feature extractor
    zoomout = Zoomout().float().to(device=device)
    for param in zoomout.parameters():
        param.requires_grad = False

    # Load data, extract features, save features
    for split in ('train', 'val', 'test'):
        save_features(args, zoomout, split)


if __name__ == '__main__':
    main()
