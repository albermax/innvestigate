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

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import RMSprop

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.tests.networks.base
import innvestigate.utils.visualizations as ivis


base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))


keras.backend.set_image_data_format("channels_first")


###############################################################################
###############################################################################
###############################################################################


def fetch_data():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1, 28, 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test


def create_model():
    num_classes = 10

    network = innvestigate.utils.tests.networks.base.mlp_2dense(
        (None, 1, 28, 28),
        num_classes,
        dense_units=1024,
        dropout_rate=0.25)
    model_wo_sm = Model(inputs=network["in"], outputs=network["out"])
    model_w_sm = Model(inputs=network["in"], outputs=network["sm_out"])
    return model_wo_sm, model_w_sm


def train_model(model, data):
    batch_size = 128
    num_classes = 10
    epochs = 20

    x_train, y_train, x_test, y_test = data
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    pass

###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":

    zero_mean = False
    pattern_type = "linear"
    #pattern_type = "relu"
    data = fetch_data()
    images = [(data[2][i].copy(), data[3][i]) for i in range(10)]
    label_to_class_name = [str(i) for i in range(10)]

    ###########################################################################
    # Utility function.
    ###########################################################################

    def preprocess(X):
        X.copy()
        X /= 255
        if zero_mean:
            X -= 0.5
        return X

    def postprocess(X):
        X = X.copy()
        X = ivis.postprocess_images(X,
                                    channels_first=False)
        return X

    def image(X):
        X = X.copy()
        X = ivis.postprocess_images(X,
                                    channels_first=False)
        return ivis.graymap(X,
                            input_is_postive_only=True)

    def bk_proj(X):
        return ivis.graymap(X)

    def heatmap(X):
        return ivis.heatmap(X)

    def graymap(X):
        return ivis.graymap(np.abs(X), input_is_postive_only=True)

    ###########################################################################
    # Build model.
    ###########################################################################
    data_preprocessed = (preprocess(data[0]), data[1],
                         preprocess(data[2]), data[3])
    model, modelp = create_model()
    train_model(modelp, data_preprocessed)
    model.set_weights(modelp.get_weights())

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
        ("pattern.net",           {},                       bk_proj, "PatterNet"),

        # Interaction
        ("pattern.attribution",   {},                       heatmap, "PatternAttribution"),
        ("lrp.z_baseline",        {},                       heatmap, "LRP-Z"),
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
                           row_label_offset=5,
                           col_label_offset=15,
                           usetex=True, file_name="mnist_all_methods.pdf")
