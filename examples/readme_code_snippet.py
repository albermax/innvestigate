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


###############################################################################
###############################################################################
###############################################################################


import imp
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))


###############################################################################
###############################################################################
###############################################################################


if __name__ == "__main__":
    # Load an image.
    # Need to download examples images first.
    # See script in images directory.
    image = utils.load_image(
        os.path.join(base_dir, "images", "ILSVRC2012_val_00011670.JPEG"), 224)

    # Code snippet.
    plt.imshow(image/255)
    plt.axis('off')
    plt.savefig("readme_example_input.png")

    import innvestigate
    import innvestigate.utils
    import keras.applications.vgg16 as vgg16

    # Get model
    model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
    # Strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.create_analyzer("deep_taylor", model)

    # Add batch axis and preprocess
    x = preprocess(image[None])
    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    # Plot
    plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
    plt.axis('off')
    plt.savefig("readme_example_analysis.png")
