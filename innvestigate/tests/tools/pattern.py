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


from keras.datasets import mnist
import keras.layers
import keras.models
from keras.models import Model
import keras.optimizers
import numpy as np
import unittest

# todo:fix relative imports:
#from ...utils.tests import dryrun
from innvestigate.utils.tests import dryrun

import innvestigate
from innvestigate.tools import PatternComputer


###############################################################################
###############################################################################
###############################################################################


class TestPatterComputer_dummy_parallel(dryrun.PatternComputerTestCase):

    def _method(self, model):
        return PatternComputer(model, pattern_type="dummy",
                               compute_layers_in_parallel=True)


class TestPatterComputer_dummy_sequential(dryrun.PatternComputerTestCase):

    def _method(self, model):
        return PatternComputer(model, pattern_type="dummy",
                               compute_layers_in_parallel=False)


###############################################################################
###############################################################################
###############################################################################


class TestPatterComputer_linear(dryrun.PatternComputerTestCase):

    def _method(self, model):
        return PatternComputer(model, pattern_type="linear")


class TestPatterComputer_relupositive(dryrun.PatternComputerTestCase):

    def _method(self, model):
        return PatternComputer(model, pattern_type="relu.positive")


class TestPatterComputer_relunegative(dryrun.PatternComputerTestCase):

    def _method(self, model):
        return PatternComputer(model, pattern_type="relu.negative")


###############################################################################
###############################################################################
###############################################################################


class HaufePatternExample(unittest.TestCase):

    def test(self):
        np.random.seed(234354346)
        # need many samples to get close to optimum and stable numbers
        n = 1000

        a_s = np.asarray([1, 0]).reshape((1, 2))
        a_d = np.asarray([1, 1]).reshape((1, 2))
        y = np.random.uniform(size=(n, 1))
        eps = np.random.rand(n, 1)

        X = y * a_s + eps * a_d

        model = keras.models.Sequential(
            [keras.layers.Dense(1, input_shape=(2,), use_bias=True), ]
        )
        model.compile(optimizer=keras.optimizers.Adam(lr=1), loss="mse")
        history = model.fit(X, y, epochs=20, verbose=0).history
        #print(history)
        self.assertTrue(model.evaluate(X, y, verbose=0) < 0.05)

        pc = PatternComputer(model, pattern_type="linear")
        A = pc.compute(X)[0]
        W = model.get_weights()[0]

        #print(a_d, model.get_weights()[0])
        #print(a_s, A)

        def allclose(a, b):
            return np.allclose(a, b, rtol=0.05, atol=0.05)

        # perpendicular to a_d
        self.assertTrue(allclose(a_d.ravel(), abs(W.ravel())))
        # estimated pattern close to true pattern
        self.assertTrue(allclose(a_s.ravel(), A.ravel()))


###############################################################################
###############################################################################
###############################################################################


def fetch_data():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.reshape(60000, 1, 28, 28) - 127.5) / 127.5
    x_test = (x_test.reshape(10000, 1, 28, 28) - 127.5) / 127.5
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train[:6000], y_train[:6000], x_test, y_test


def create_model(clazz):
    num_classes = 10

    network = clazz(
        (None, 1, 28, 28),
        num_classes,
        dense_units=1024,
        dropout_rate=0.25)
    model_wo_sm = Model(inputs=network["in"], outputs=network["out"])
    model_w_sm = Model(inputs=network["in"], outputs=network["sm_out"])
    return model_wo_sm, model_w_sm


def train_model(model, data, epochs=20):
    batch_size = 128
    num_classes = 10

    x_train, y_train, x_test, y_test = data
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    pass


class MnistPatternExample_dense_linear(unittest.TestCase):

    def test(self):
        np.random.seed(234354346)
        model_class = innvestigate.utils.tests.networks.base.mlp_2dense

        data = fetch_data()
        model, modelp = create_model(model_class)
        train_model(modelp, data, epochs=1)
        model.set_weights(modelp.get_weights())

        analyzer = innvestigate.create_analyzer("pattern.net", model)
        analyzer.fit(data[0], pattern_type="linear",
                     batch_size=256, verbose=1)

        patterns = analyzer._patterns
        W = model.get_weights()[0]
        W2D = W.reshape((-1, W.shape[-1]))
        X = data[0].reshape((data[0].shape[0], -1))
        Y = np.dot(X, W2D)

        def safe_divide(a, b):
            return a / (b + (b == 0))

        mean_x = X.mean(axis=0)
        mean_y = Y.mean(axis=0)
        mean_xy = np.dot(X.T, Y) / Y.shape[0]
        ExEy = mean_x[:, None] * mean_y[None, :]
        cov_xy = mean_xy - ExEy
        w_cov_xy = np.diag(np.dot(W2D.T, cov_xy))
        A = safe_divide(cov_xy, w_cov_xy[None, :])

        def allclose(a, b):
            return np.allclose(a, b, rtol=0.05, atol=0.05)
        #print(A.sum(), patterns[0].sum())
        self.assertTrue(allclose(A.ravel(), patterns[0].ravel()))


class MnistPatternExample_dense_relu(unittest.TestCase):

    def test(self):
        np.random.seed(234354346)
        model_class = innvestigate.utils.tests.networks.base.mlp_2dense

        data = fetch_data()
        model, modelp = create_model(model_class)
        train_model(modelp, data, epochs=1)
        model.set_weights(modelp.get_weights())

        analyzer = innvestigate.create_analyzer("pattern.net", model)
        analyzer.fit(data[0], pattern_type="relu",
                     batch_size=256, verbose=1)
        patterns = analyzer._patterns
        W, b = model.get_weights()[:2]
        W2D = W.reshape((-1, W.shape[-1]))
        X = data[0].reshape((data[0].shape[0], -1))
        Y = np.dot(X, W2D)

        mask = np.dot(X, W2D) + b > 0
        count = mask.sum(axis=0)

        def safe_divide(a, b):
            return a / (b + (b == 0))

        mean_x = safe_divide(np.dot(X.T, mask), count)
        mean_y = Y.mean(axis=0)
        mean_xy = safe_divide(np.dot(X.T, Y * mask), count)

        ExEy = mean_x * mean_y

        cov_xy = mean_xy - ExEy
        w_cov_xy = np.diag(np.dot(W2D.T, cov_xy))
        A = safe_divide(cov_xy, w_cov_xy[None, :])

        def allclose(a, b):
            return np.allclose(a, b, rtol=0.05, atol=0.05)
        #print(A.sum(), patterns[0].sum())
        self.assertTrue(allclose(A.ravel(), patterns[0].ravel()))
