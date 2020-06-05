# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 06: train_segmentation.py
# Description:
#   -

import os
import datetime
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from data.loader import xView2, visualize
from nets.zoomout import Zoomout
from nets.classifier import FCClassifier, DenseClassifier
from utils.class_utils import label_accuracy_score

best_acc = 0

def train(args, zoomout, model, train_loader, optimizer, loss_fn, device, epoch):

    # Initialize output path
    OUT_PATH = os.path.join(os.sep, "scratch", "jonathantan", "results")

    # Setup
    model.train()
    model = model.to(device=device)
    zoomout = zoomout.to(device=device)
    count = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        # Print batch number every 1000 batches
        if batch_idx % 1000 == 0:
            print(datetime.datetime.now(), "\t Batch", batch_idx + 1, "of", len(train_loader),
                    end="\t")
        # move to cuda
        if torch.cuda.is_available():
            images = images.to(device=device)
            labels = labels.to(device=device)

        images, labels = images.float(), labels.float()

        # Downsample and extract zoomout features
        N, C, H, W = images.shape
        # downsample = nn.Conv2d(C, C, (1, 1), stride=2, padding=0)
        with torch.no_grad():
            zoom_feats = zoomout.forward(images)

        # Main forward and backward pass
        optimizer.zero_grad()
        if args.batch_size == 1:
            y_pred = model.forward(zoom_feats.unsqueeze(0))
        else:
            y_pred = model.forward(zoom_feats)
        del zoom_feats
        loss = loss_fn(y_pred, labels.long())
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            count = count + 1
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.item()))

        if batch_idx % 1000 == 0:
            """
            Visualization of results.
            """
            gt = labels[0,:,:].detach().cpu().numpy().squeeze().astype(int)
            im = images[0,:,:,:].detach().cpu().numpy().squeeze()
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            pred = y_pred[0,:,:,:].cpu()
            _, pred_mx = torch.max(pred, 0)
            pred = pred_mx.detach().cpu().numpy().squeeze().astype(int)
            image = Image.fromarray(im.astype(np.uint8), mode='RGB')

            # Save images and predictions
            img_stub = str(epoch).zfill(2) + "_" + str(count).zfill(3)
            image.save(os.path.join(OUT_PATH, img_stub + ".png"))
            visualize(os.path.join(OUT_PATH, img_stub + "_pred.png"), pred)
            visualize(os.path.join(OUT_PATH, img_stub + "_gt.png"), gt)

            del gt, im, pred, _, pred_mx, image

        del images, labels, y_pred

    # Make sure to save your model periodically
    print("Saving model")
    torch.save(model, os.path.join("models", "full_model.pkl"))


def val(args, zoomout, model, val_loader, device):

    global best_acc
    global best_model

    model.eval()
    model = model.to(device=device)
    zoomout = zoomout.to(device=device)
    print(datetime.datetime.now(), "Validating...")
    label_trues, label_preds = [], []

    for batch_idx, (data, target) in enumerate(val_loader):

        if batch_idx % 1000 == 0:
            print(datetime.datetime.now(), "\t Batch", batch_idx + 1, "of", len(val_loader))

        # move to cuda
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data, target = data.float(), target.float()

        N, C, H, W = data.shape
        with torch.no_grad():
            zoom_feats = zoomout.forward(data)
            if args.batch_size == 1:
                y_score = model.forward(zoom_feats.unsqueeze(0))
            else:
                y_score = model.forward(zoom_feats)
            del zoom_feats

            _, pred = torch.max(y_score, 1)
            lbl_pred = pred.detach().cpu().numpy().astype(np.int64)
            lbl_true = target.detach().cpu().numpy().astype(np.int64)

            for _, lt, lp in zip(_, lbl_true, lbl_pred):
                label_trues.append(lt)
                label_preds.append(lp)

    n_class = 2
    metrics = label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    \tAccuracy: {0}
    \tAccuracy Class: {1}
    \tMean IU: {2}
    \tFWAV Accuracy: {3}'''.format(*metrics))

    # Save best model
    acc = metrics[0]
    if acc > best_acc:
        best_model = model
        best_acc = acc


def main():

    np.random.seed(seed=0)

    # Initialize batch job arguments
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='./models/best_fc_cls.pkl', help='Path to the saved model')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,    help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=64,  help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, help='Learning Rate')
    args = parser.parse_args()

    # Setup GPU settings
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Training using", device)

    # Setup feature extractor; no learning required
    zoomout = Zoomout().float()
    for param in zoomout.parameters():
        param.requires_grad = False

    # Setup classifier and optimizer
    fc_classifier = torch.load(args.model_path)
    classifier = DenseClassifier(fc_model=fc_classifier, device=device).float()
    optimizer = optim.Adam(classifier.parameters(), lr=args.l_rate)

    # Setup datasets
    dataset_train = xView2(split='train')
    dataset_test = xView2(split='test')

    # Use inverse class weights for loss function
    inv_counts = torch.Tensor(
       np.load(os.path.join(os.sep, "scratch", "jonathantan", "features", "inv_counts.npy"))
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=inv_counts)
    del inv_counts

    # Wrap data in dataloader classes
    train_loader = data.DataLoader(dataset_train,
                                   batch_size=args.batch_size,
                                   num_workers=0,
                                   shuffle=True)
    test_loader = data.DataLoader(dataset_test,
                                  batch_size=args.batch_size,
                                  num_workers=0,
                                  shuffle=False)
    del dataset_train, dataset_test

    # Epoch loop
    best_acc = 0
    for epoch in range(args.n_epoch):
        train(args, zoomout, classifier, train_loader, optimizer, loss_fn, device, epoch)
        val(args, zoomout, classifier, test_loader, device)

    # Save best model
    SAVE_PATH = os.path.join("models", "best_model_dict.pt")
    torch.save(best_model.state_dict(), SAVE_PATH)
    print("Validation with best model:")
    val(args, zoomout, best_model, test_loader, device)

if __name__ == '__main__':
    main()
