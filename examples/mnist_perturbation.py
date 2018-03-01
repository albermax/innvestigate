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

from innvestigate.tools import Perturbation


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
    epochs = 1

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
                        verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score

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
    score = train_model(modelp, data_preprocessed)
    model.set_weights(modelp.get_weights())

    ###########################################################################
    # Analysis.
    ###########################################################################
    perturbation_function = np.zeros_like  # TODO

    # Create analyzers.
    method = ("lrp.z_baseline", {}, heatmap, "LRP-Z")
    analyzer = innvestigate.create_analyzer(method[0],
                                            model,
                                            **method[1])
    analyzer.fit(data_preprocessed[0], pattern_type=pattern_type,
                 batch_size=256, verbose=0)
    # Create analysis.
    x_test, y_test = data_preprocessed[2:]
    x_test_perturbated = x_test  # TODO inplace?
    batch_size = 256
    current_index = 0
    while current_index < x_test.shape[0]:
        if current_index + batch_size < x_test.shape[0]:
            curr_batch_size = batch_size
        else:
            current_batch_size = x_test.shape[0] - current_index
        image = x_test[current_index:current_index + curr_batch_size]
        # Predict label.
        x = preprocess(image)
        perturbation = Perturbation(analyzer, perturbation_function, steps=1)
        a = analyzer.analyze(x)
        x_perturbated = perturbation.compute_on_batch(x, a)
        current_index += batch_size
    num_classes = 10
    score_perturbated = modelp.evaluate(x_test_perturbated, keras.utils.to_categorical(y_test, num_classes), verbose=0)
    print("score {}, score_perturbated {}".format(score, score_perturbated))
