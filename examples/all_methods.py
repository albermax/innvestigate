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


import matplotlib

matplotlib.use('Agg')

import imp
import keras.backend
import keras.models
import matplotlib.pyplot as plt
import numpy as np
import os

import innvestigate
import innvestigate.utils.tests.networks.imagenet
import innvestigate.utils.visualizations as ivis


keras.backend.set_image_data_format("channels_first")


###############################################################################
###############################################################################
###############################################################################


# todo: make nicer!


def load_parameters(filename):
    f = np.load(filename)
    ret = [f["arr_%i" % i] for i in range(len(f.keys()))]
    return ret


def load_patterns(filename):
    f = np.load(filename)

    ret = {}
    for prefix in ["A", "r", "mu"]:
        l = sum([x.startswith(prefix) for x in f.keys()])
        ret.update({prefix: [f["%s_%i" % (prefix, i)] for i in range(l)]})

    return ret


def lasagne_weights_to_keras_weights(weights):
    ret = []
    for w in weights:
        if len(w.shape) < 4:
            ret.append(w)
        else:
            ret.append(w.transpose(2, 3, 1, 0))
    return ret


###############################################################################
###############################################################################
###############################################################################


base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))

param_file = "./imagenet_224_vgg_16.npz"
# Note those weights are CC 4.0:
# See http://www.robots.ox.ac.uk/~vgg/research/very_deep/
param_url = "https://www.dropbox.com/s/cvjj8x19hzya9oe/imagenet_224_vgg_16.npz?dl=1"

pattern_file = "./imagenet_224_vgg_16.pattern_file.A_only.npz"
pattern_url = "https://www.dropbox.com/s/v7e0px44jqwef5k/imagenet_224_vgg_16.patterns.A_only.npz?dl=1"


###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":
    # Download the necessary parameters for VGG16 and the according patterns.
    eutils.download(param_url, param_file)
    eutils.download(pattern_url, pattern_file)

    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data()[:2]


    ###########################################################################
    # Build model.
    ###########################################################################
    parameters = lasagne_weights_to_keras_weights(load_parameters(param_file))
    vgg16 = innvestigate.utils.tests.networks.imagenet.vgg16()
    model = keras.models.Model(inputs=vgg16["in"], outputs=vgg16["out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.set_weights(parameters)
    modelp = keras.models.Model(inputs=vgg16["in"], outputs=vgg16["sm_out"])
    modelp.compile(optimizer="adam", loss="categorical_crossentropy")
    modelp.set_weights(parameters)

    ###########################################################################
    # Utility function.
    ###########################################################################

    def preprocess(X):
        X = X.copy()
        X = ivis.preprocess_images(X, color_coding="RGBtoBGR")
        X = innvestigate.utils.tests.networks.imagenet.vgg16_preprocess(X)
        return X

    def postprocess(X):
        X = X.copy()
        return X

    def image(X):
        X = innvestigate.utils.tests.networks.imagenet.vgg16_invert_preprocess(X)
        X = ivis.postprocess_images(X,
                                    color_coding="BGRtoRGB",
                                    channels_first=False)
        X = ivis.project(X, absmax=255.0, input_is_postive_only=True)
        return X

    def bk_proj(X):
        X = ivis.postprocess_images(X,
                                    color_coding="BGRtoRGB",
                                    channels_first=False)
        return ivis.project(X)

    def heatmap(X):
        X = ivis.postprocess_images(X,
                                    color_coding="BGRtoRGB",
                                    channels_first=False)
        return ivis.heatmap(X)

    ###########################################################################
    # Analysis.
    ###########################################################################

    # Methods we use and some properties.
    methods = [
        # NAME             POSTPROCESSING     TITLE

        # Show input.
        ("input",               image,   "Input"),

        # Function
        ("gradient",            bk_proj, "Gradient"),

        # Signal
        ("deconvnet",           bk_proj, "Deconvnet"),
        ("guided_backprop",     bk_proj, ("Guided", "Backprop"),),
        ("pattern.net",         bk_proj, "PatterNet"),

        # Interaction
        ("pattern.attribution", heatmap, "PatternAttribution"),
        ("lrp.z_baseline",      heatmap, "LRP-Z"),
    ]

    # Create analyzers.
    patterns = lasagne_weights_to_keras_weights(
        load_patterns(pattern_file)["A"])
    analyzers = []
    for method in methods:
        kwargs = {}
        if method[0].startswith("pattern"):
            kwargs["patterns"] = patterns
        analyzers.append(innvestigate.create_analyzer(method[0],
                                                      model,
                                                      **kwargs))

    # Create analysis.
    analysis = np.zeros([len(images), len(analyzers), 224, 224, 3])
    text = []
    for i, (image, y) in enumerate(images):
        # Predict label.
        x = preprocess(image[None, :, :, :])
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        text.append((r"\textbf{%s}" % label_to_class_name[y],
                     r"\textit{(%.2f)}" % prob.max(),
                     r"\textit{%s}" % label_to_class_name[y_hat]))

        for aidx, analyzer in enumerate(analyzers):
            # Analyze.
            a = analyzer.analyze(x)
            # Postprocess.
            a = postprocess(a)
            a = methods[aidx][1](a)
            analysis[i, aidx] = a[0]

    ###########################################################################
    # Plot the analysis.
    ###########################################################################

    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    row_labels = text
    col_labels = [method[2] for method in methods]

    eutils.plot_image_grid(grid, row_labels, col_labels,
                           row_label_offset=50,
                           col_label_offset=-50,
                           usetex=True, file_name="all_methods.pdf")
