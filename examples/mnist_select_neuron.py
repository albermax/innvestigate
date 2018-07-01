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
import keras.backend
import keras.models
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


    # Analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS               POSTPROC FXN            TITLE

        # Show input.
        ("input",                 {},                       mutils.image,          "Input"),

        # Function
        ("gradient",              {},                       mutils.graymap,        "Gradient"),

        # Signal
        ("deconvnet",             {},                       mutils.bk_proj,        "Deconvnet"),
        ("guided_backprop",       {},                       mutils.bk_proj,        "Guided Backprop",),
        ("pattern.net",           {},                       mutils.bk_proj,        "PatternNet"),

        # Interaction
        ("lrp.z",                 {},                       mutils.heatmap,         "LRP-Z"),
        ("lrp.epsilon",           {"epsilon": 1},           mutils.heatmap,         "LRP-Epsilon"),
    ]



    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],
                                                model,
                                                neuron_selection_mode="index",
                                                **method[1])
        analyzer.fit(data_preprocessed[0], pattern_type=pattern_type,
                     batch_size=256, verbose=1)

        analyzers.append(analyzer)




    ###########################################################################
    # Analysis  fixed input image, iterate over output neurons.
    ###########################################################################
    # Select and fix image index for analysis
    input_image_idx = 1
    image, y = images[input_image_idx]
    image = image[None, :, :, :]

    # Create analysis.
    analysis = np.zeros([10, len(analyzers), 28, 28, 3])
    text = []
    for i in range(10):
        print ('Output Neuron {}: '.format(i), end='')
        # Predict label.

        x = mutils.preprocess(image, input_range) #prediction is the same for every iteration below
        presm = model.predict_on_batch(x)[0]
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()


        text.append((r"%s" % label_to_class_name[y],
                     r"%.2f" % presm[i],
                     r"%.2f" % prob[i],
                     r"%s" % label_to_class_name[i]))

        for aidx, analyzer in enumerate(analyzers):
            #measure execution time
            t_start = time.time()
            print('{} '.format(methods[aidx][-1]), end='')

            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            if is_input_analyzer:
                a = analyzer.analyze(image, i)
            else:
                a = analyzer.analyze(x, i)


            t_elapsed = time.time() - t_start
            print('({:.4f}s) '.format(t_elapsed), end='')

            # Postprocess.
            if not is_input_analyzer:
                a = mutils.postprocess(a)
            a = methods[aidx][2](a)
            analysis[i, aidx] = a[0]
        print('')

    ###########################################################################
    # Plot this analysis.
    ###########################################################################

    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]), 'neuron: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]

    eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                           file_name="mnist_fixed_input_image_{}_{}.pdf".format(input_image_idx, modelname))


    #clean shutdown for tf.
    if K.backend() == 'tensorflow':
        K.clear_session()
