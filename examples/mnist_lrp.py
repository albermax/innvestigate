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
import time

import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import RMSprop, SGD, Adam

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.tests.networks.base
import innvestigate.utils.visualizations as ivis
import innvestigate.applications
import innvestigate.applications.mnist


base_dir = os.path.dirname(__file__)
#import utils.py as eutils and utils_mnist.py as mutils in non-module environment
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))
mutils = imp.load_source("utils_mnist", os.path.join(base_dir, "utils_mnist.py"))


###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":
    # A list of predefined models. Set modelname to one of the keys in dict models below.
    modelname = 'pretrained_plos_long_relu'

    #Adapt to Play around!
    #         MODELNAME                      DATA INPUT RANGE           TRAINING EPOCHS     MODEL CREATION KWARGS
    models = {'mlp_2dense':                  ([-1, 1],                  15,                 {'dense_units':1024, 'dropout_rate':0.25, 'activation':'relu'}),
              'mlp_3dense':                  ([-1, 1],                  20,                 {'dense_units':1024, 'dropout_rate':0.25}),
              'cnn_2convb_2dense':           ([-.5, .5],                20,                 {}),

              # pre-trained model from [https://doi.org/10.1371/journal.pone.0130140 , http://jmlr.org/papers/v17/15-618.html]
              'pretrained_plos_long_relu':   ([-1, 1],                  0,                  {}),
              'pretrained_plos_short_relu':  ([-1, 1],                  0,                  {}),
              'pretrained_plos_long_tanh':   ([-1, 1],                  0,                  {}),
              'pretrained_plos_short_tanh':  ([-1, 1],                  0,                  {}),
             }

    # unpack model params by name
    input_range, n_epochs, kwargs = models[modelname]




    ###########################################################################
    # Get Data / Set Parameters
    ###########################################################################
    pattern_type = "relu"
    channels_first = keras.backend.image_data_format == "channels_first"
    data = mutils.fetch_data(channels_first)
    images = [(data[2][i].copy(), data[3][i]) for i in range(10)]
    label_to_class_name = [str(i) for i in range(10)]



    ###########################################################################
    # Build model.
    ###########################################################################
    data_preprocessed = (mutils.preprocess(data[0], input_range), data[1],
                         mutils.preprocess(data[2], input_range), data[3])
    model, modelp = mutils.create_model(channels_first, modelname, **kwargs)
    mutils.train_model(modelp, data_preprocessed, epochs=n_epochs)
    model.set_weights(modelp.get_weights())




    ###########################################################################
    # Analysis.
    ###########################################################################



    # Methods we use and some properties.
    methods = [
        # NAME                    OPT.PARAMS               POSTPROC FXN            TITLE

        # Show input.
        ("input",                 {},                       mutils.image,          "Input"),

        # Function
        ("gradient",              {},                       mutils.graymap,        "Gradient"),
        ("smoothgrad",            {"noise_scale": 50},      mutils.graymap,        "SmoothGrad"),
        ("integrated_gradients",  {},                       mutils.graymap,        "Integrated Gradients"),

        # Signal
        ("deconvnet",             {},                       mutils.bk_proj,        "Deconvnet"),
        ("guided_backprop",       {},                       mutils.bk_proj,        "Guided Backprop",),
        ("pattern.net",           {},                       mutils.bk_proj,        "PatterNet"),

        # Interaction
        ("lrp.z_baseline",        {},                       mutils.heatmap,         "Gradient*Input"),
        ("lrp.z",                 {},                       mutils.heatmap,         "LRP-Z"),
        ("lrp.epsilon",           {"epsilon": 1},           mutils.heatmap,         "LRP-Epsilon"),
        ("lrp.composite_a",       {},                       mutils.heatmap,         "LRP-CompositeA"),
        ("lrp.composite_b",       {"epsilon": 1},           mutils.heatmap,         "LRP-CompositeB"),
    ]

    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],
                                                model,
                                                **method[1])
        analyzer.fit(data_preprocessed[0], pattern_type=pattern_type,
                     batch_size=256, verbose=1)

        analyzers.append(analyzer)

    # Create analysis.
    analysis = np.zeros([len(images), len(analyzers), 28, 28, 3])
    text = []
    for i, (image, y) in enumerate(images):
        print ('Image {}: '.format(i), end='', flush=True)
        image = image[None, :, :, :]
        # Predict label.
        x = mutils.preprocess(image, input_range)
        presm = model.predict_on_batch(x)[0]
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        text.append((r"%s" % label_to_class_name[y],
                     r"%.2f" % presm.max(),
                     r"(%.2f)" % prob.max(),
                     r"%s" % label_to_class_name[y_hat]))

        for aidx, analyzer in enumerate(analyzers):
            #measure execution time
            t_start = time.time()
            print('{} '.format(methods[aidx][-1]), end='', flush=True)

            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            a = analyzer.analyze(image if is_input_analyzer else x)

            t_elapsed = time.time() - t_start
            print('({:.4f}s) '.format(t_elapsed), end='', flush=True)

            # Postprocess.
            if not is_input_analyzer:
                a = mutils.postprocess(a)
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
                           row_label_offset=5,
                           col_label_offset=15,
                           usetex=False,
                           is_fontsize_adaptive=False,
                           file_name="mnist_all_methods_{}.pdf".format(modelname))

    #clean shutdown for tf.
    if K.backend() == 'tensorflow':
        K.clear_session()
