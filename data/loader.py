# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 01: loader.py
# Description:
#  - Extends pytorch Dataset class that holds xView2 training and test data.
#  - Ingests images and labels in .png format and defines __getitem__ func
#       that returns one image and one label in numpy array form.
#  - Randomly chooses half of the dataset to be test data on init.

import os
from PIL import Image
import torch
import numpy as np
from torch.utils import data
from imageio import imwrite

class xView2(data.Dataset):

    def __init__(self, root='/scratch/jonathantan/cropped/images', split='train'):
        self.root = root
        self.split = split
        self.im_height = 512
        self.im_width = 512

        # Populate list of image filepaths, pre-disaster only
        image_paths = os.listdir(self.root)

        # Split into train and validation sets
        image_paths = np.array(image_paths)
        num_train = int(len(image_paths) // 2)
        train_idx = np.random.choice(range(len(image_paths)), size=num_train)
        test_mask = np.ones(len(image_paths), np.bool)
        test_mask[train_idx] = 0
        self.files = {
            'train': image_paths[train_idx],
            'test': image_paths[test_mask]
        }

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[self.split][index])
        lbl_path = img_path.replace('images', 'labels')

        # Open, process, reshape, convert to numpy array
        pil_img = Image.open(img_path).convert('RGB')
        img = np.array(pil_img).swapaxes(0, 2).swapaxes(1, 2)

        # Generate labels for missing masks
        if os.path.exists(lbl_path):
            pil_lbl = Image.open(lbl_path).convert("P")
            lbl = np.array(pil_lbl)
            lbl[lbl != 0] = 1 # Change "True" values in label from 225 to 1
        else:
            lbl = np.zeros((self.im_height, self.im_width))

        # Return correct datatype
        if torch.cuda.is_available():
            dtype = torch.cuda.ByteTensor
        else:
            dtype = torch.ByteTensor

        return torch.from_numpy(img).type(dtype), torch.from_numpy(lbl).type(dtype)


def visualize(path, predicted_label):

    # Colors for labels
    colors = np.asarray([[0,0,0], [128,0,0]])

    # Initialize empty image to color
    label = predicted_label
    label_viz = np.zeros((label.shape[0], label.shape[1], 3))

    # Loop through unique labels and assign unique color to empty image
    for unique_class in np.unique(label):
        if unique_class != 0:
            indices = np.argwhere(label==unique_class)
            for idx in range(indices.shape[0]):
                label_viz[indices[idx, 0], indices[idx, 1], :] = colors[unique_class, :]

    # Save image
    label_viz_uint8 = label_viz.astype(np.uint8)
    imwrite(path, label_viz_uint8)
