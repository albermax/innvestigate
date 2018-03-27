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
import keras.backend as K
import keras.models
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

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

    netname = sys.argv[1] if len(sys.argv) > 1 else "vgg16"
    pattern_type = "relu"

    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data()[:2]


    ###########################################################################
    # Build model.
    ###########################################################################
    tmp = getattr(innvestigate.applications.imagenet, netname)
    # todo: specify type of patterns:
    net = tmp(load_weights=True, load_patterns=pattern_type)
    model = keras.models.Model(inputs=net["in"], outputs=net["out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    modelp = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
    modelp.compile(optimizer="adam", loss="categorical_crossentropy")

    ###########################################################################
    # Utility functions.
    ###########################################################################
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
    channels_first = K.image_data_format == "channels_first"

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
        #X = ivis.gamma(X, minamp=0)
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

        # Interaction
        ("lrp.z_baseline",        {},                       heatmap, "Gradient*Input"),
        ("lrp.z",                 {},                       heatmap, "LRP-Z"),
        ("lrp.z_IB",              {},                       heatmap, "LRP-Z-IB"),
        ("lrp.epsilon",           {},                       heatmap, "LRP-Epsilon"),
        ("lrp.epsilon_IB",        {},                       heatmap, "LRP-Epsilon-IB"),
        ("lrp.w_square",          {},                       heatmap, "LRP-W-Square"),
        ("lrp.flat",              {},                       heatmap, "LRP-Flat"),
        #("lrp.alpha_beta",        {},                       heatmap, "LRP-AB"),
        ("lrp.alpha_2_beta_1",    {},                       heatmap, "LRP-A2B1"),
        ("lrp.alpha_2_beta_1_IB", {},                       heatmap, "LRP-A2B1-IB"),
        ("lrp.alpha_1_beta_0",    {},                       heatmap, "LRP-A1B0"),
        ("lrp.alpha_1_beta_0_IB", {},                       heatmap, "LRP-A1B0-IB"),
        ("lrp.z_plus",            {},                       heatmap, "LRP-ZPlus"),
        ("lrp.z_plus_fast",       {},                       heatmap, "LRP-ZPlusFast"),
        ("lrp.composite_a",           {},                     heatmap, "LRP-CompositeA"),
        ("lrp.composite_b",           {},                     heatmap, "LRP-CompositeB")
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
        print ('Image {}: '.format(i), end='')
        image = image[None, :, :, :]
        # Predict label.
        x = preprocess(image)
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        text.append((r"%s" % label_to_class_name[y],
                     r"(%.2f)" % prob.max(),
                     r"%s" % label_to_class_name[y_hat]))

        for aidx, analyzer in enumerate(analyzers):
            #measure execution time
            t_start = time.time()
            print('{} '.format(methods[aidx][-1]), end='')

            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            a = analyzer.analyze(image if is_input_analyzer else x)

            t_elapsed = time.time() - t_start
            print('({:.4f}s) '.format(t_elapsed), end='')

            # Postprocess.
            if not np.all(np.isfinite(a)):
                print("Image %i, analysis of %s not finite: nan %s inf %s" %
                      (i, methods[aidx][3],
                       np.any(np.isnan(a)), np.any(np.isinf(a))))
            if not is_input_analyzer:
                a = postprocess(a)
            a = methods[aidx][2](a)
            analysis[i, aidx] = a[0]
        print('')

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
                           usetex=False,
                           is_fontsize_adaptive=False,
                           file_name="imagenet_lrp_%s.pdf" % netname)

    #clean shutdown for tf.
    if K.backend() == 'tensorflow':
        K.clear_session()
