# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 02: zoomout.py
# Description:
#  - Defines hypercolumn feature extractor; passes images through pretrained
#       VGG11 model and concatenates feature activations at chosen layers.
#  - Default activations are from layers directly preceding pooling layers.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

class Zoomout(nn.Module):

    def __init__(self):
        super(Zoomout, self).__init__()

        # load the pre-trained ImageNet CNN and normalize per
        # https://pytorch.org/docs/stable/torchvision/models.html
        self.vgg = models.vgg16(pretrained=True)
        self.feature_list = list(self.vgg.features.children())
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Get indices of Conv2d layers to extract activations from
        # vgg11 conv layers before maxpool layers
        # self.zoomout_layers = [0, 3, 8, 13, 18]
        # vgg16 conv layers before maxpool layers
        # self.zoomout_layers = [2, 7, 14, 21, 28]

    def forward(self, x):

        # Get original H and W to upsample to
        N, C, H, W = x.shape

        # Compress to [0, 1] and normalize
        x = x.float() / 255
        x = self.norm(x.squeeze(0)).unsqueeze(0)

        # Loop over activation functions and store results of selected layers
        activations = []
        activations.append(x)

        for idx, layer in enumerate(self.feature_list):

            # Append upsampled layer activations
            x = layer(x)
            if isinstance(layer, nn.Conv2d):

                # Upsample to original H and W using bilinear interpolation
                upsampled = F.interpolate(x,
                                          size=(H, W),
                                          mode='bilinear',
                                          align_corners=True)

                # Store and later concatenate
                activations.append(upsampled)

        return torch.cat(activations, dim=1).squeeze(0)
