# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from keras.datasets import mnist
import keras.layers
import keras.models
from keras.models import Model
import keras.optimizers
import numpy as np
import unittest

from innvestigate.utils.tests import dryrun

import innvestigate
from innvestigate.tools import PatternComputer


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternComputer_dummy_parallel():

    def method(model):
        return PatternComputer(model, pattern_type="dummy",
                               compute_layers_in_parallel=True)

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.skip("Feature not supported.")
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternComputer_dummy_sequential():

    def method(model):
        return PatternComputer(model, pattern_type="dummy",
                               compute_layers_in_parallel=False)

    dryrun.test_pattern_computer(method, "mnist.log_reg")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternComputer_linear():

    def method(model):
        return PatternComputer(model, pattern_type="linear")

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PatternComputer_linear():

    def method(model):
        return PatternComputer(model, pattern_type="linear")

    dryrun.test_pattern_computer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternComputer_relupositive():

    def method(model):
        return PatternComputer(model, pattern_type="relu.positive")

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PatternComputer_relupositive():

    def method(model):
        return PatternComputer(model, pattern_type="relu.positive")

    dryrun.test_pattern_computer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternComputer_relunegative():

    def method(model):
        return PatternComputer(model, pattern_type="relu.negative")

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PatternComputer_relunegative():

    def method(model):
        return PatternComputer(model, pattern_type="relu.negative")

    dryrun.test_pattern_computer(method, "mnist.*")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
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
        model.fit(X, y, epochs=20, verbose=0).history
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

    return x_train[:100], y_train[:100], x_test[:10], y_test[:10]


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

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0)
    model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)


@pytest.mark.fast
@pytest.mark.precommit
class MnistPatternExample_dense_linear(unittest.TestCase):

    def test(self):
        np.random.seed(234354346)
        model_class = innvestigate.utils.tests.networks.base.mlp_2dense

        data = fetch_data()
        model, modelp = create_model(model_class)
        train_model(modelp, data, epochs=10)
        model.set_weights(modelp.get_weights())

        analyzer = innvestigate.create_analyzer("pattern.net", model,
                                                pattern_type="linear")
        analyzer.fit(data[0], batch_size=256, verbose=0)

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


@pytest.mark.fast
@pytest.mark.precommit
class MnistPatternExample_dense_relu(unittest.TestCase):

    def test(self):
        np.random.seed(234354346)
        model_class = innvestigate.utils.tests.networks.base.mlp_2dense

        data = fetch_data()
        model, modelp = create_model(model_class)
        train_model(modelp, data, epochs=10)
        model.set_weights(modelp.get_weights())

        analyzer = innvestigate.create_analyzer("pattern.net", model,
                                                pattern_type="relu")
        analyzer.fit(data[0], batch_size=256, verbose=0)
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


# def extract_2d_patches(X, conv_layer):

#     X_in = X
#     kernel_shape = conv_layer.kernel_size
#     strides = conv_layer.strides
#     rates = conv_layer.dilation_rate
#     padding = conv_layer.padding

#     assert all([x == 1 for x in rates])
#     assert all([x == 3 for x in kernel_shape])
#     assert all([x == 1 for x in strides])

#     if padding.lower() == "same":
#         tmp = np.ones(list(X.shape[:2])+[x+3 for x in X.shape[2:]],
#                       dtype=X.dtype)
#         tmp[:, :, 1:-2, 1:-2] = X
#         X = tmp

#     out_shape = [int(np.ceil((x-k)/s))
#                  for x, k, s in zip(X.shape[2:], kernel_shape, strides)]
#     n_patches = np.prod(list(X.shape[:2])+out_shape)
#     dimensions = X.shape[1]*kernel_shape[0]*kernel_shape[1]
#     ret = np.empty((n_patches, dimensions), dtype=X.dtype)

#     i_ret = 0
#     for j in range(X.shape[2]-kernel_shape[0]):
#         for k in range(X.shape[3]-kernel_shape[1]):
#             patches = X[:, :, j:j+kernel_shape[0], k:k+kernel_shape[1]]
#             patches = patches.reshape((-1, dimensions))
#             ret[i_ret:i_ret+X.shape[0]] = patches
#             i_ret += X.shape[0]

#     if True:
#         import tensorflow as tf
#         with tf.Session():
#             tf_ret = tf.extract_image_patches(
#                 images=X_in.transpose((0, 2, 3, 1)),
#                 ksizes=[1, kernel_shape[0], kernel_shape[1], 1],
#                 strides=[1, strides[0], strides[1], 1],
#                 rates=[1, rates[0], rates[1], 1],
#                 padding=padding.upper()).eval()

#         tf_ret = tf_ret.reshape((-1, tf_ret.shape[-1]))
#         #print(tf_ret.shape, ret.shape)
#         assert tf_ret.shape == ret.shape
#         #print(tf_ret.mean(), ret.mean())
#         assert tf_ret.mean() == ret.mean()
#     assert i_ret == n_patches
#     return ret


# class __disabled__MnistPatternExample_conv_linear(unittest.TestCase):

#     def test(self):
#         np.random.seed(234354346)
#         K.set_image_data_format("channels_first")
#         model_class = innvestigate.utils.tests.networks.base.cnn_2convb_2dense
#         data = fetch_data()
#         model, modelp = create_model(model_class)
#         train_model(modelp, data, epochs=1)
#         model.set_weights(modelp.get_weights())
#         analyzer = innvestigate.create_analyzer("pattern.net", model)
#         analyzer.fit(data[0], pattern_type="linear",
#                      batch_size=256, verbose=0)

#         patterns = analyzer._patterns
#         W = model.get_weights()[0]
#         W2D = W.reshape((-1, W.shape[-1]))
#         X = extract_2d_patches(data[0], model.layers[1])
#         Y = np.dot(X, W2D)

#         def safe_divide(a, b):
#             return a / (b + (b == 0))

#         mean_x = X.mean(axis=0)
#         mean_y = Y.mean(axis=0)
#         mean_xy = np.dot(X.T, Y) / Y.shape[0]

#         ExEy = mean_x[:, None] * mean_y[None, :]
#         cov_xy = mean_xy - ExEy
#         w_cov_xy = np.diag(np.dot(W2D.T, cov_xy))
#         A = safe_divide(cov_xy, w_cov_xy[None, :])

#         def allclose(a, b):
#             return np.allclose(a, b, rtol=0.05, atol=0.05)

#         self.assertTrue(allclose(A.ravel(), patterns[0].ravel()))
