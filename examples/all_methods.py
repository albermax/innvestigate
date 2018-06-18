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
import keras.backend as K
import keras.models
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
in_utils = imp.load_source("utils", os.path.join(base_dir, "utils_imagenet.py"))

###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":

    # pass a model name from innvestigate.applications.imagenet via command line
    netname = sys.argv[1] if len(sys.argv) > 1 else "vgg16"
    pattern_type = "relu"

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
    # Handle Input.
    ###########################################################################
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
    channels_first = keras.backend.image_data_format == "channels_first"


    ###########################################################################
    # Analysis.
    ###########################################################################
    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data(
        net["image_shape"][0])

    patterns = net["patterns"]
    # Methods we use and some properties.
    methods = [
        # NAME                    OPT.PARAMS               POSTPROC FXN             TITLE

        # Show input.
        ("input",                 {},                       in_utils.image,         "Input"),

        # Function
        ("gradient",              {},                       in_utils.graymap,       "Gradient"),
        ("smoothgrad",            {"noise_scale": 50},      in_utils.graymap,       "SmoothGrad"),
        ("integrated_gradients",  {},                       in_utils.graymap,       "Integrated Gradients"),

        # Signal
        ("deconvnet",             {},                       in_utils.bk_proj,       "Deconvnet"),
        ("guided_backprop",       {},                       in_utils.bk_proj,       "Guided Backprop",),
        ("pattern.net",           {"patterns": patterns},   in_utils.bk_proj,       "PatternNet"),

        # Interaction
        ("pattern.attribution",   {"patterns": patterns},   in_utils.heatmap,       "PatternAttribution"),
        ("lrp.z_baseline",        {},                       in_utils.heatmap,       "Gradient*Input"),
        ("lrp.z",                 {},                       in_utils.heatmap,       "LRP-Z"),
        ("lrp.epsilon",           {"epsilon": 1},           in_utils.heatmap,       "LRP-Epsilon"),
        ("lrp.sequential_preset_a_flat", {"epsilon": 1},         in_utils.heatmap,         "LRP-PresetAFlat"),
        ("lrp.sequential_preset_b_flat", {"epsilon": 1},         in_utils.heatmap,         "LRP-PresetBFlat"),
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
    analysis = np.zeros([len(images), len(analyzers)]+net["image_shape"]+[3])
    text = []
    for i, (image, y) in enumerate(images):
        print ('Image {}: '.format(i), end='', flush=True)
        image = image[None, :, :, :]
        # Predict label.
        x = in_utils.preprocess(image, net)
        presm = model.predict_on_batch(x)[0]
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        text.append((r"%s" % label_to_class_name[y],
                     r"%.2f" % presm.max(),
                     r"%.2f" % prob.max(),
                     r"%s" % label_to_class_name[y_hat]))

        for aidx, analyzer in enumerate(analyzers):
            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            if analyzer:
                # Measure execution time
                t_start = time.time()
                print('{} '.format(methods[aidx][-1]), end='', flush=True)

                a = analyzer.analyze(image if is_input_analyzer else x)

                t_elapsed = time.time() - t_start
                print('({:.4f}s) '.format(t_elapsed), end='', flush=True)

                # Postprocess.
                if not np.all(np.isfinite(a)):
                    print("Image %i, analysis of %s not finite: nan %s inf %s" %
                          (i, methods[aidx][3],
                           np.any(np.isnan(a)), np.any(np.isinf(a))))
                if not is_input_analyzer:
                    a = in_utils.postprocess(a, color_conversion, channels_first)
                a = methods[aidx][2](a)
            else:
                a = np.zeros_like(image)
            analysis[i, aidx] = a[0]
        print('')

    ###########################################################################
    # Plot the analysis.
    ###########################################################################

    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]), 'pred: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]

    eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                           file_name="all_methods_{}.pdf".format(netname))

    #clean shutdown for tf.
    if K.backend() == 'tensorflow':
        K.clear_session()
