from __future__ import annotations

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.backend import image_data_format
from tensorflow.keras.datasets import mnist

import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis

###############################################################################
# Data Preprocessing Utility
###############################################################################


def fetch_data():
    channels_first = image_data_format() == "channels_first"
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


def create_preprocessing_f(X, input_range=None):
    """
    Generically shifts data from interval [a, b] to interval [c, d].
    Assumes that theoretical min and max values are populated.
    """
    if input_range is None:
        input_range = [0, 1]

    if len(input_range) != 2:
        raise ValueError(f"Input range must be of length 2, but was {len(input_range)}")
    if input_range[0] >= input_range[1]:
        raise ValueError(
            f"Values in input_range must be ascending. It is {input_range}"
        )

    a, b = X.min(), X.max()
    c, d = input_range

    def preprocessing(X):
        # Make sure images have shape (28, 28, 1) and cast from uint8 to float32
        X = np.expand_dims(X, -1).astype(np.float32)
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


def train_model(model, data, batch_size=128, epochs=20):
    num_classes = 10

    x_train, y_train, x_test, y_test = data
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
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
