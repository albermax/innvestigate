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
import sys
import os

import innvestigate
import innvestigate.utils.tests.networks.imagenet
import innvestigate.utils.visualizations as ivis


###############################################################################
###############################################################################
###############################################################################


base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))


###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":

    netname = sys.argv[1] if len(sys.argv) > 1 else "vgg16"
    pattern_type = "relu"

    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data()[:2]


    ###########################################################################
    # Build model.
    ###########################################################################
    tmp = getattr(innvestigate.utils.tests.networks.imagenet, netname)
    net = tmp(weights="imagenet")
    model = keras.models.Model(inputs=net["in"], outputs=net["out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    modelp = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
    modelp.compile(optimizer="adam", loss="categorical_crossentropy")

    patterns_file = np.load(
        "%s_patterns_type_%s_tf_dim_ordering_tf_kernels.npz" %
        (pattern_type, netname))
    patterns = [patterns_file["arr_%i" % i]
                for i in range(len((patterns_file.keys())))]

    print("\n".join([str((x.min(), x.mean(), x.max())) for x in patterns]))
    ###########################################################################
    # Utility functions.
    ###########################################################################
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
    channels_first = keras.backend.image_data_format == "channels_first"

    def preprocess(X):
        X = X.copy()
        X = net["preprocess_f"](X)
        return X

    def postprocess(X):
        X = X.copy()
        X = ivis.postprocess_images(X,
                                    color_coding=color_conversion,
                                    channels_first=channels_first)
        return X

    def image(X):
        X = X.copy()
        return ivis.project(X, absmax=255.0, input_is_postive_only=True)

    def bk_proj(X):
        X = ivis.clip_quantile(X, 1)
        return ivis.project(X)

    def heatmap(X):
        return ivis.heatmap(X)

    def graymap(X):
        return ivis.graymap(np.abs(X), input_is_postive_only=True)

    ###########################################################################
    # Analysis.
    ###########################################################################

    # Methods we use and some properties.
    methods = [
        # NAME             POSTPROCESSING     TITLE

        # Show input.
        ("input",                 {},                       image,   "Input"),

        # Function
        ("gradient",              {},                       graymap, "Gradient"),
        ("smoothgrad",            {"noise_scale": 50},      graymap, "SmoothGrad"),
        ("integrated_gradients",  {},                       graymap, ("Integrated", "Gradients")),

        # Signal
        ("deconvnet",             {},                       bk_proj, "Deconvnet"),
        ("guided_backprop",       {},                       bk_proj, ("Guided", "Backprop"),),
        ("pattern.net",           {"patterns": patterns},   bk_proj, "PatterNet"),

        # Interaction
        ("pattern.attribution",   {"patterns": patterns},   heatmap, "PatternAttribution"),
        ("lrp.z_baseline",        {},                       heatmap, "LRP-Z"),
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
            if not np.all(np.isfinite(a)):
                print("Image %i, analysis of %s not finite: nan %s inf %s" %
                      (i, methods[aidx][3],
                       np.any(np.isnan(a)), np.any(np.isinf(a))))
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
                           usetex=True, file_name="all_methods.pdf")
