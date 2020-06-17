# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File 06: train_segmentation.py
# Description:
#   -

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

from data.loader import xView2, visualize
from nets.zoomout import Zoomout
from nets.classifier import FCClassifier, DenseClassifier
from utils.class_utils import label_accuracy_score

best_acc = 0

def train(args, zoomout, model, train_loader, optimizer, loss_fn, device, epoch):

    print("Training Epoch [%d/%d]" % (epoch+1, args.n_epoch))

    # Setup
    model.train()
    model = model.to(device=device)
    losses = []
    img_count = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        # move to cuda
        images, labels = images.float(), labels.float()
        if torch.cuda.is_available():
            images = images.to(device=device)
            labels = labels.to(device=device)

        # Downsample and extract zoomout features
        N, C, H, W = images.shape
        with torch.no_grad():
            zoom_feats = zoomout.forward(images)

        # Forward pass
        optimizer.zero_grad()
        if args.batch_size == 1:
            y_score = model.forward(zoom_feats.unsqueeze(0))
        else:
            y_score = model.forward(zoom_feats)
        del zoom_feats

        # Backward pass
        loss = loss_fn(y_score, labels.long())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            """
            Visualization of results.
            """
            gt = labels[0,:,:].detach().clone().cpu().numpy().squeeze().astype(int)
            im = images[0,:,:,:].detach().clone().cpu().numpy().squeeze()
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            pred = y_score[0,:,:,:].cpu()
            _, pred_mx = torch.max(pred, 0)
            pred = pred_mx.detach().cpu().numpy().squeeze().astype(int)
            image = Image.fromarray(im.astype(np.uint8), mode='RGB')

            # Save images and predictions
            img_stub = str(epoch).zfill(2) + "_" + str(img_count).zfill(3)
            image.save(os.path.join(args.out_path, img_stub + ".png"))
            visualize(os.path.join(args.out_path, img_stub + "_pred.png"), pred)
            visualize(os.path.join(args.out_path, img_stub + "_gt.png"), gt)
            img_count += 1

            del gt, im, pred, _, pred_mx, image

        del images, labels, y_score

    # Print average training loss across all batches
    mean_loss = np.mean(losses)
    print("\tMean training loss:", mean_loss)

    # Save model periodically
    print("\tSaving model for this epoch")
    torch.save(model.state_dict(), os.path.join("models", "most_recent_model.pt"))

    return mean_loss


def test(args, zoomout, model, val_loader, loss_fn, device, val=True):

    global best_acc
    global best_model

    # Setup
    model.eval()
    model = model.to(device=device)
    if val:
        print("\tValidating...")
    label_trues, label_preds = [], []
    losses = []

    # Iterate over batches of data
    for batch_idx, (images, labels) in enumerate(val_loader):

        # move to cuda
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        images, labels = images.float(), labels.float()

        N, C, H, W = images.shape
        with torch.no_grad():
            zoom_feats = zoomout.forward(images)
            if args.batch_size == 1:
                y_score = model.forward(zoom_feats.unsqueeze(0))
            else:
                y_score = model.forward(zoom_feats)
            del zoom_feats

            _, pred = torch.max(y_score, 1)
            lbl_pred = pred.detach().clone().cpu().numpy().astype(np.int64)
            lbl_true = labels.detach().clone().cpu().numpy().astype(np.int64)
            loss = loss_fn(y_score, labels.long())
            losses.append(loss.item())

            for _, lt, lp in zip(_, lbl_true, lbl_pred):
                label_trues.append(lt)
                label_preds.append(lp)

    # Print performance metrics
    n_class = 2
    metrics = label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    \tAccuracy: {0}
    \tAccuracy Class: {1}
    \tMean IU: {2}
    \tFWAV Accuracy: {3}'''.format(*metrics))

    # Print average training loss across all batches
    mean_loss = np.mean(losses)
    if test:
        print('Test loss:', mean_loss)
    else:
        print('\tValidation loss:', mean_loss)

    # Save best model
    acc = metrics[0]
    if val and acc > best_acc:
        best_model = model
        best_acc = acc

    return mean_loss


def plot_learning_curve(args, train_losses, val_losses):

    epochs = [i + 1 for i in range(args.n_epoch)]
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.legend()
    plt.savefig("learning_curve_seg.png")

    return None


def main():

    np.random.seed(seed=0)

    # Initialize batch job arguments
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--load_saved_model', nargs='?', type=bool, default=False, help='If true, loads existing most_recent_model.pt')
    parser.add_argument('--model_path', nargs='?', type=str, default='./models', help='Path to the saved models')
    parser.add_argument('--feature_path', nargs='?', type=str, default='/scratch/jonathantan/cv/features', help='Path to the saved hypercol features')
    parser.add_argument('--out_path', nargs='?', type=str, default='/scratch/jonathantan/cv/results', help='Directory to save results to')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=3,    help='# of epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,  help='Batch size')
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

    # Setup feature extractor; no learning required
    zoomout = Zoomout().float().to(device=device)
    for param in zoomout.parameters():
        param.requires_grad = False

    # Load fully connected classifier
    fc_classifier = FCClassifier(device=device, n_hidden=args.n_hidden).float()
    fc_classifier.load_state_dict(torch.load(os.path.join(args.model_path, "best_fc_dict.pt")))

    # Load previous model state if pretrained
    classifier = DenseClassifier(fc_model=fc_classifier, device=device, n_hidden=args.n_hidden).float()
    saved_model_path = os.path.join(args.model_path, "most_recent_model.pt")
    if args.load_saved_model and os.path.exists(saved_model_path):
        print("Loading saved state and continuing training...")
        classifier.load_state_dict(torch.load(saved_model_path))

    # Set up optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=args.l_rate)

    # Use inverse class weights for loss function - true distribution is 0.95-0.05
    WEIGHTS = torch.Tensor([0.05, 0.95]).to(device=device)
    loss_fn = nn.CrossEntropyLoss(weight=WEIGHTS)
    del WEIGHTS

    # Setup datasets
    dataset_train = xView2(split='train')
    dataset_val = xView2(split='val')
    dataset_test = xView2(split='test')

    # Wrap data in dataloader classes
    train_loader = data.DataLoader(dataset_train,
                                   batch_size=args.batch_size,
                                   num_workers=0,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset_val,
                                  batch_size=args.batch_size,
                                  num_workers=0,
                                  shuffle=False)
    test_loader = data.DataLoader(dataset_test,
                                  batch_size=args.batch_size,
                                  num_workers=0,
                                  shuffle=False)
    del dataset_train, dataset_val, dataset_test

    # Load saved model performance if available
    saved_losses_path = os.path.join(args.model_path, "most_recent_model_state.npy")
    if args.load_saved_model and os.path.exists(saved_losses_path):
        train_losses, val_losses = np.load(saved_losses_path)
        train_losses, val_losses = train_losses.tolist(), val_losses.tolist()
    else:
        train_losses, val_losses = [], []

    # Main training loop
    start_epoch = len(train_losses)
    for epoch in range(start_epoch, start_epoch + args.n_epoch):
        train_loss = train(args, zoomout, classifier, train_loader, optimizer, loss_fn, device, epoch)
        val_loss = test(args, zoomout, classifier, val_loader, loss_fn, device, val=True)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Run best model on test set
    print("Validation with best model:")
    test(args, zoomout, best_model, test_loader, loss_fn, device, val=False)

    # Export learning curve and save current loss performance
    plot_learning_curve(args, train_losses, val_losses)
    np.save(os.path.join(args.model_path, "most_recent_model_state.npy"), (train_losses, val_losses))

    # Save best model
    SAVE_PATH = os.path.join(args.model_path, "best_model_dict.pt")
    torch.save(best_model.state_dict(), SAVE_PATH)


if __name__ == '__main__':
    main()
