# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import shutil


###############################################################################
# Download utilities
###############################################################################


def download(url, filename):
    if not os.path.exists(filename):
        print("Download: %s ---> %s" % (url, filename))
        with six.moves.urllib.request.urlopen(url) as response:
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

###############################################################################
# Plot utility
###############################################################################


def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    return ret


def get_imagenet_data():
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "images", "ground_truth")) as f:
        ground_truth = {x.split()[0]: int(x.split()[1])
                        for x in f.readlines() if len(x.strip()) > 0}
    with open(os.path.join(base_dir, "images", "imagenet_label_mapping")) as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}

    images = [(load_image(os.path.join(base_dir, "images", f), 224),
               ground_truth[f])
              for f in os.listdir(os.path.join(base_dir, "images"))
              if f.endswith(".JPEG")]

    return images, image_label_mapping


def plot_image_grid(grid,
                    row_labels,
                    col_labels,
                    file_name=None,
                    row_label_offset=0,
                    col_label_offset=0,
                    usetex=False,
                    size_per_cell=3,
                    dpi=224):
    n_rows = len(grid)
    n_cols = len(grid[0])
    shape_per_image = grid[0][0].shape[:2]
    n_padding = shape_per_image[0]//5
    shape_per_image_padded = [s + 2 * n_padding for s in shape_per_image]
    fontsize = shape_per_image[1]//2

    plt.clf()
    plt.figure(figsize=(n_rows * size_per_cell,
                        n_cols * size_per_cell),
               dpi=dpi)
    plt.tick_params(axis="x", which="both",
                    bottom="off", top="off", labelbottom="off")
    plt.tick_params(axis="y", which="both",
                    bottom="off", top="off", labelbottom="off")
    plt.axis("off")
    plt.rc("text", usetex=usetex)
    plt.rc("font", family="sans-serif")

    # Plot grid.
    image_grid = np.ones((n_rows * shape_per_image_padded[0],
                          n_cols * shape_per_image_padded[1],
                          3),
                         dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] is not None:
                pos = (i*shape_per_image_padded[0]+n_padding,
                       j*shape_per_image_padded[1]+n_padding)
                image_grid[pos[0]:pos[0]+shape_per_image[0],
                           pos[1]:pos[1]+shape_per_image[1], :] = grid[i][j]

    plt.imshow(image_grid, interpolation="nearest")

    # Plot the row labels.
    for i, label in enumerate(row_labels):
        if not isinstance(label, (list, tuple)):
            label = (label,)
        for j, s in enumerate(label):
            plt.text(0,
                     row_label_offset+
                     n_padding +
                     shape_per_image_padded[1] * i +
                     shape_per_image[1] * j / len(label),
                     s, fontsize=fontsize, ha="right")

    # Plot the col labels.
    for i, label in enumerate(col_labels):
        if not isinstance(label, (list, tuple)):
            label = (label,)
        for j, s in enumerate(label):
            plt.text(n_padding + shape_per_image_padded[1] * i,
                     col_label_offset - shape_per_image[1] +
                     shape_per_image[1] * j / len(label),
                     s, fontsize=fontsize, ha="left")

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
