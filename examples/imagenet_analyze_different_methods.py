# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import \
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imp
import time
import sys
import os
import argparse

import keras.backend
import keras.models

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.tests.networks.imagenet
import innvestigate.utils.visualizations as ivis

base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))
imgnetutils = imp.load_source("utils", os.path.join(base_dir, "utils_imagenet.py"))

#############################################################################################
#############################################################################################

if __name__ == "__main__":

    # Parameter (run f.ex. as imagenet_analyze_different_methods.py 0 --nets vgg16 vgg19)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('integer', type=int, help='a number between 0-6 to choose for an example image')
    parser.add_argument('--nets', nargs='+', help='list of netnames')
    args = parser.parse_args()

    image_n = args.integer
    netnames = args.nets

    n_nets = len(netnames)

    # Analysis

    pattern_type = "relu"
    channels_first = keras.backend.image_data_format == "channels_first"
    analysis_all = []

    for i, netname in enumerate(netnames):
        print("Analyse {}.".format(netname))

        # Build model.
        tmp = getattr(innvestigate.applications.imagenet, netname)
        net = tmp(load_weights=True, load_patterns=pattern_type)
        model = keras.models.Model(inputs=net["in"], outputs=net["out"])
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        modelp = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
        modelp.compile(optimizer="adam", loss="categorical_crossentropy")

        # Load image from example folder
        images, label_to_class_name = eutils.get_imagenet_data(net["image_shape"][0])
        images = images[image_n]
        label_to_class_name = label_to_class_name[image_n]

        color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None

        # Analysis.
        patterns = net["patterns"]

        methods = [
            # NAME             POSTPROCESSING     TITLE

            # Show input.
            ("input", {}, imgnetutils.image, "Input"),

            # Function
            ("gradient", {}, imgnetutils.graymap, "Gradient"),
            ("integrated_gradients", {}, imgnetutils.graymap, ("Integrated", "Gradients")),

            # Signal
            ("deconvnet", {}, imgnetutils.bk_proj, "Deconvnet"),
            ("guided_backprop", {}, imgnetutils.bk_proj, ("Guided", "Backprop"),),
            ("pattern.net", {"patterns": patterns}, imgnetutils.bk_proj, "PatterNet"),

            # Interaction
            ("pattern.attribution", {"patterns": patterns}, imgnetutils.heatmap, ("Pattern", "Attribution"),),
            ("lrp.epsilon", {}, imgnetutils.heatmap, "LRP Epsilon"),
            ("lrp.sequential_preset_a", {}, imgnetutils.heatmap, ("LRP Sequential", "Preset A")),
            ("lrp.sequential_preset_b", {}, imgnetutils.heatmap, ("LRP Sequential", "Preset B"))
        ]

        # Create analyzers.
        analyzers = []
        for method in methods:
            try:
                analyzer = innvestigate.create_analyzer(method[0],
                                                        model,
                                                        **method[1])
            except innvestigate.NotAnalyzeableModelException:
                analyzer = None
            analyzers.append(analyzer)

        # Create analysis.
        analysis = np.zeros([len(analyzers)] + net["image_shape"] + [3])

        image, y = images
        image = image[None, :, :, :]
        # Predict label.
        x = imgnetutils.preprocess(image, net)
        presm = model.predict_on_batch(x)[0]
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        for aidx, analyzer in enumerate(analyzers):
            # Measure execution time
            t_start = time.time()
            print('{} '.format(''.join(methods[aidx][-1])), end='')

            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            if analyzer != None:
                a = analyzer.analyze(image if is_input_analyzer else x)
            else:
                print("Analyzer not available for this model.")
                a = np.zeros_like(x)

            t_elapsed = time.time() - t_start
            print('({:.4f}s) '.format(t_elapsed), end='')

            # Postprocess.
            if not np.all(np.isfinite(a)):
                print("Image %i, analysis of %s not finite: nan %s inf %s" %
                      (i, methods[aidx][3],
                       np.any(np.isnan(a)), np.any(np.isinf(a))))
            if not is_input_analyzer:
                a = imgnetutils.postprocess(a, color_conversion, channels_first)
            a = methods[aidx][2](a)
            analysis[aidx] = a[0]

            if analyzer != None:
                print('Finished analysis of {}'.format(''.join(methods[aidx][-1])))

        # Clear session.
        if keras.backend.backend() == 'tensorflow':
            keras.backend.clear_session()

        analysis_all.append(analysis)

    # Plot the analysis.
    grid = [[analysis_all[i][j] for j in range(len(methods))]
                 for i in range(n_nets)]
    row_labels_left = row_labels_left = [(n,'') for n in netnames]
    col_labels = [''.join(method[3]) for method in methods]

    eutils.plot_image_grid(grid, row_labels_left, [], col_labels, file_name="different_methods.pdf")