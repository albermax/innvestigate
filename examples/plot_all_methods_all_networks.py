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
import sys
import os
import argparse

import innvestigate
import innvestigate.utils as iutils
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

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('integer', type=int, help='a number between 0-6 to choose for an example image')
    parser.add_argument('--nets', nargs='+',  help='list of netnames')
    args = parser.parse_args()

    image_n = args.integer
    netnames = args.nets
    n_nets = len(netnames)
    n_analyses = 9

    pattern_type = "relu"

    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data()[:2]
    # only select one image
    images = images[image_n]
    label_to_class_name = label_to_class_name[image_n]

    analysis = np.zeros([n_nets, n_analyses, 224, 224, 3])

    for i, netname in enumerate(netnames):
        print("Analyse {}.".format(netname))
        ###########################################################################
        # Build model.
        ###########################################################################
        tmp = getattr(innvestigate.applications.imagenet, netname)
        net = tmp(load_weights=True, load_patterns=pattern_type)
        model = keras.models.Model(inputs=net["in"], outputs=net["out"])
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        modelp = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
        modelp.compile(optimizer="adam", loss="categorical_crossentropy")

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
            X = iutils.postprocess_images(X,
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

        patterns = net["patterns"]
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

        text = []
        image,y = images
        image = image[None, :, :, :]
        # Predict label.
        x = preprocess(image)
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()


        for aidx, analyzer in enumerate(analyzers):
            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            a = analyzer.analyze(image if is_input_analyzer else x)
            # Postprocess.
            if not np.all(np.isfinite(a)):
                print("Analysis of %s not finite: nan %s inf %s" %
                      (methods[aidx][3], np.any(np.isnan(a)), np.any(np.isinf(a))))
            if not is_input_analyzer:
                a = postprocess(a)
            a = methods[aidx][2](a)
            analysis[i, aidx] = a[0]

        ###########################################################################
        # Plot the analysis.
        ###########################################################################

        grid = [[analysis[i, j] for j in range(analysis.shape[1])]
                for i in range(n_nets)]
        row_labels = netnames
        col_labels = [method[3] for method in methods]

        eutils.plot_image_grid(grid, row_labels, col_labels,
                               row_label_offset=50,
                               col_label_offset=-50,
                               usetex=True, file_name="all_methods_all_nets.pdf")
