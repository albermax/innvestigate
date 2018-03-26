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

import imp
import keras.backend
import keras.models
import matplotlib.pyplot as plt
import numpy as np
import os

import innvestigate
import innvestigate.utils.tests.networks.imagenet
import innvestigate.utils.visualizations as ivis


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

    keras.backend.set_image_data_format("channels_first")

    ###########################################################################
    # Build model.
    ###########################################################################
    parameters = lasagne_weights_to_keras_weights(load_parameters(param_file))
    vgg16 = innvestigate.utils.tests.networks.imagenet.vgg16_custom()
    model = keras.models.Model(inputs=vgg16["in"], outputs=vgg16["out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.set_weights(parameters)
    modelp = keras.models.Model(inputs=vgg16["in"], outputs=vgg16["sm_out"])
    modelp.compile(optimizer="adam", loss="categorical_crossentropy")
    modelp.set_weights(parameters)

    patterns = lasagne_weights_to_keras_weights(
        load_patterns(pattern_file)["A"])

    ###########################################################################
    # Utility functions.
    ###########################################################################

    def preprocess(X):
        X = X.copy()
        X = ivis.preprocess_images(X, color_coding="RGBtoBGR")
        X = innvestigate.utils.tests.networks.imagenet.vgg16_custom_preprocess(X)
        return X

    def postprocess(X):
        X = X.copy()
        X = ivis.postprocess_images(X,
                                    color_coding="BGRtoRGB",
                                    channels_first=channels_first)
        return X

    def image(X):
        X = X.copy()
        return ivis.project(X, absmax=255.0, input_is_postive_only=True)

    def heatmap(X):
        return ivis.heatmap(X)

    ###########################################################################
    # Analysis.
    ###########################################################################

    # Methods we use and some properties.
    methods = [
        # NAME             POSTPROCESSING     TITLE

        # Show input.
        ("input",                 {},                       image,   "Input"),

        # Interaction
        ("lrp.z_baseline",        {},                       heatmap, "LRP-Z Baseline"),
        ("lrp.z",                 {},                       heatmap, "LRP-Z"),
        ("lrp.z_WB",              {},                       heatmap, "LRP-Z-WB"),
        ("lrp.z_plus",            {},                       heatmap, "LRP-ZPlus"),
        ("lrp.epsilon",           {},                       heatmap, "LRP-Epsilon"),
        ("lrp.epsilon_WB",        {},                       heatmap, "LRP-Epsilon-WB"),
        ("lrp.w_square",          {},                       heatmap, "LRP-W-Square"),
        ("lrp.flat",              {},                       heatmap, "LRP-Flat"),
        ("lrp.alpha_1_beta_1",    {},                       heatmap, "LRP-A1B1"),
        ("lrp.alpha_1_beta_1_WB", {},                       heatmap, "LRP-A1B1-WB"),
        ("lrp.alpha_2_beta_1",    {},                       heatmap, "LRP-A2B1"),
        ("lrp.alpha_2_beta_1_WB", {},                       heatmap, "LRP-A2B1-WB"),
        ("lrp.alpha_1_beta_0",    {},                       heatmap, "LRP-A1B0"),
        ("lrp.alpha_1_beta_0_WB", {},                       heatmap, "LRP-A1B0-WB"),
    ]

    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzers.append(innvestigate.create_analyzer(method[0],
                                                      model,
                                                      **method[1]))

    # Create analysis.
    analysis = np.zeros([len(images), len(analyzers), 224, 224, 3])
    text = []
    for i, (image, y) in enumerate(images):
        image = image[None, :, :, :]
        # Predict label.
        x = preprocess(image)
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        text.append((r"\textbf{%s}" % label_to_class_name[y],
                     r"\textit{(%.2f)}" % prob.max(),
                     r"\textit{%s}" % label_to_class_name[y_hat]))

        for aidx, analyzer in enumerate(analyzers):
            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            a = analyzer.analyze(image if is_input_analyzer else x)
            # Postprocess.
            if not is_input_analyzer:
                a = postprocess(a)
            a = methods[aidx][2](a)
            analysis[i, aidx] = a[0]

    ###########################################################################
    # Plot the analysis.
    ###########################################################################

    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    row_labels = text
    col_labels = [method[3] for method in methods]

    eutils.plot_image_grid(grid, row_labels, col_labels,
                           row_label_offset=50,
                           col_label_offset=-50,
                           usetex=True, file_name="all_methods_lrp.pdf")
