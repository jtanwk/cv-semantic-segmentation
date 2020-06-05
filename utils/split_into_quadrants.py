import os
import itertools
import numpy as np
from PIL import Image


def open_and_crop(path, idx, factor, output, set='images'):

    # Check which set
    if set == 'labels':
        path = path.replace("images", "masks")

    # Initialize cropped image size
    size = 1024 // factor

    if os.path.exists(path):

        # Generate coordinate grid and define crop boundaries
        starts = np.arange(0, 1024, size) # start, stop, step
        grid = np.array(list(itertools.product(starts, starts)))

        # Loop over grid and crop image
        img = Image.open(path)
        FILE_PATH = os.path.join(output, set, str(idx).zfill(5))
        for n, i in enumerate(grid):
            h, w = i[0], i[1]
            cropped = img.crop((h, w, h + size, w + size))
            cropped.save(FILE_PATH + "_" + str(n + 1).zfill(2) + ".png")

    else:

        lbl = Image.fromarray(np.zeros((size, size))).convert('RGB')
        FILE_PATH = os.path.join(output, set, str(idx).zfill(5))
        for n in range(factor ** 2):
            lbl.save(FILE_PATH + "_" + str(n + 1).zfill(2) + ".png")


def crop_images(input, output, factor=4):

    # Get list of filenames, pre-disaster only
    disasters = os.listdir(input)
    image_paths = []
    for i in disasters:
        if i == '.DS_Store':
            continue
        file_path = os.path.join(input, i, 'images')
        file_list = os.listdir(file_path)
        file_list_pre = [x for x in file_list if "pre" in x]
        file_list_pre_full = [os.path.join(file_path, x) for x in file_list_pre]
        image_paths.extend(file_list_pre_full)

    for idx, p in enumerate(image_paths):
        print(idx, "/", len(image_paths))
        open_and_crop(p, idx+1, factor, output, 'images')
        open_and_crop(p, idx+1, factor, output, 'labels')



if __name__ == '__main__':

    # Setup paths
    input_path = './data/xBD'
    output_path = './data/cropped'

    crop_images(input_path, output_path)
