# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import absolute_import, division, print_function, unicode_literals

# catch exception with: except Exception as e
from builtins import filter, map, range, zip
from io import open

import keras
import numpy as np
import six
from future.utils import raise_from, raise_with_traceback
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam

import innvestigate
import innvestigate.applications.mnist
import innvestigate.utils
import innvestigate.utils as iutils
import innvestigate.utils.tests
import innvestigate.utils.tests.networks
import innvestigate.utils.visualizations as ivis

# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


###############################################################################
# Data Preprocessing Utility
###############################################################################


def fetch_data():
    channels_first = K.image_data_format() == "channels_first"
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if channels_first:
        x_train = x_train.reshape(60000, 1, 28, 28)
        x_test = x_test.reshape(10000, 1, 28, 28)
    else:
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    return x_train, y_train, x_test, y_test


def create_preprocessing_f(X, input_range=[0, 1]):
    """
    Generically shifts data from interval [a, b] to interval [c, d].
    Assumes that theoretical min and max values are populated.
    """

    if len(input_range) != 2:
        raise ValueError(
            "Input range must be of length 2, but was {}".format(len(input_range))
        )
    if input_range[0] >= input_range[1]:
        raise ValueError(
            "Values in input_range must be ascending. It is {}".format(input_range)
        )

    a, b = X.min(), X.max()
    c, d = input_range

    def preprocessing(X):
        # shift original data to [0, b-a] (and copy)
        X = X - a
        # scale to new range gap [0, d-c]
        X /= b - a
        X *= d - c
        # shift to desired output range
        X += c
        return X

    def revert_preprocessing(X):
        X = X - c
        X /= d - c
        X *= b - a
        X += a
        return X

    return preprocessing, revert_preprocessing


############################
# Model Utility
############################


def create_model(modelname, **kwargs):
    channels_first = K.image_data_format() == "channels_first"
    num_classes = 10

    if channels_first:
        input_shape = (None, 1, 28, 28)
    else:
        input_shape = (None, 28, 28, 1)

    # load PreTrained models
    if modelname in innvestigate.applications.mnist.__all__:
        model_init_fxn = getattr(innvestigate.applications.mnist, modelname)
        model_wo_sm, model_w_sm = model_init_fxn(input_shape[1:])

    elif modelname in innvestigate.utils.tests.networks.base.__all__:
        network_init_fxn = getattr(innvestigate.utils.tests.networks.base, modelname)
        network = network_init_fxn(input_shape, num_classes, **kwargs)
        model_wo_sm = Model(inputs=network["in"], outputs=network["out"])
        model_w_sm = Model(inputs=network["in"], outputs=network["sm_out"])
    else:
        raise ValueError("Invalid model name {}".format(modelname))

    return model_w_sm


def train_model(model, data, batch_size=128, epochs=20):
    num_classes = 10

    x_train, y_train, x_test, y_test = data
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


############################
# Post Processing Utility
############################


def postprocess(X):
    X = X.copy()
    X = iutils.postprocess_images(X)
    return X


def bk_proj(X):
    return ivis.graymap(X)


def heatmap(X):
    return ivis.heatmap(X)


def graymap(X):
    return ivis.graymap(np.abs(X), input_is_positive_only=True)
