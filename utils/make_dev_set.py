# CMSC 25040 Introduction to Computer Vision
# Final Project
# Jonathan Tan
#
# File: make_dev_set.py
# Description:
#  - Makes a 10% and 1% sample from full data to speed up development

import os
import shutil
import argparse
import numpy as np
import pandas as pd

def main():

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Sampling parameters')
    parser.add_argument('--base_imgs', nargs='?', type=str,
        default="/scratch/jonathantan/cv/cropped/images_full",
        help='Directory for full training images')
    parser.add_argument('--base_lbls', nargs='?', type=str,
        default="/scratch/jonathantan/cv/cropped/labels_full",
        help='Directory for full training labels')
    parser.add_argument('--frac', nargs='?', type=float,
        default=0.10,
        help='Fraction of samples')
    args = parser.parse_args()

    # Sample list of files to copy to new directory
    full = os.listdir(args.base_imgs)
    num_sample = int(np.round(args.frac * len(full)))
    sample_files = np.random.choice(full, size=num_sample, replace=False)

    # Make new directories labeled by fractional percentage
    frac_stub = str(int(np.round(100 * args.frac))) + "pct"
    sample_img_dir = args.base_imgs.replace("_full", frac_stub)
    sample_lbl_dir = args.base_lbls.replace("_full", frac_stub)
    if not os.path.exists(sample_img_dir):
        print("Creating", sample_img_dir)
        os.mkdir(sample_img_dir)
    if not os.path.exists(sample_lbl_dir):
        print("Creating", sample_lbl_dir)
        os.mkdir(sample_lbl_dir)

    # Copy sampled files to new directory`
    print("Copying %d files to %s" % (num_sample, sample_img_dir))
    for file in sample_files:

        # Images
        OLD_IMG_PATH = os.path.join(args.base_imgs, file)
        shutil.copy(OLD_IMG_PATH, sample_img_dir)

        # Labels
        OLD_LBL_PATH = os.path.join(args.base_lbls, file)
        shutil.copy(OLD_LBL_PATH, sample_lbl_dir)



if __name__ == '__main__':
    main()
