from __future__ import annotations

import unittest

import numpy as np
import pytest
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels
import tensorflow.keras.optimizers as koptimizers
import tensorflow.keras.utils as kutils
from tensorflow.keras.datasets import mnist

import innvestigate
import innvestigate.utils.keras.backend as ibackend
from innvestigate.tools import PatternComputer

from tests import dryrun
from tests.networks.base import mlp_2dense


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__PatternComputer_dummy_parallel():
    def method(model):
        return PatternComputer(
            model, pattern_type="dummy", compute_layers_in_parallel=True
        )

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.skip("Feature not supported.")
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__PatternComputer_dummy_sequential():
    def method(model):
        return PatternComputer(
            model, pattern_type="dummy", compute_layers_in_parallel=False
        )

    dryrun.test_pattern_computer(method, "mnist.log_reg")


###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__PatternComputer_linear():
    def method(model):
        return PatternComputer(model, pattern_type="linear")

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_precommit__PatternComputer_linear():
    def method(model):
        return PatternComputer(model, pattern_type="linear")

    dryrun.test_pattern_computer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__PatternComputer_relupositive():
    def method(model):
        return PatternComputer(model, pattern_type="relu.positive")

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_precommit__PatternComputer_relupositive():
    def method(model):
        return PatternComputer(model, pattern_type="relu.positive")

    dryrun.test_pattern_computer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__PatternComputer_relunegative():
    def method(model):
        return PatternComputer(model, pattern_type="relu.negative")

    dryrun.test_pattern_computer(method, "mnist.log_reg")


@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_precommit__PatternComputer_relunegative():
    def method(model):
        return PatternComputer(model, pattern_type="relu.negative")

    dryrun.test_pattern_computer(method, "mnist.*")


###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
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

        model = kmodels.Sequential([klayers.Dense(1, input_shape=(2,), use_bias=True)])
        model.compile(optimizer=koptimizers.Adam(lr=1), loss="mse")
        model.fit(X, y, epochs=20, verbose=0).history
        self.assertTrue(model.evaluate(X, y, verbose=0) < 0.05)

        pc = PatternComputer(model, pattern_type="linear")
        A = pc.compute(X)[0]
        W = model.get_weights()[0]

        # print(a_d, model.get_weights()[0])
        # print(a_s, A)

        def allclose(a, b):
            return np.allclose(a, b, rtol=0.05, atol=0.05)

        # perpendicular to a_d
        self.assertTrue(allclose(a_d.ravel(), abs(W.ravel())))
        # estimated pattern close to true pattern
        self.assertTrue(allclose(a_s.ravel(), A.ravel()))


###############################################################################


def fetch_data():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.reshape(60000, 1, 28, 28) - 127.5) / 127.5
    x_test = (x_test.reshape(10000, 1, 28, 28) - 127.5) / 127.5
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    return x_train[:100], y_train[:100], x_test[:10], y_test[:10]


def train_model(model, data, epochs=20):
    batch_size = 128
    num_classes = 10

    x_train, y_train, x_test, y_test = data
    # convert class vectors to binary class matrices
    y_train = kutils.to_categorical(y_train, num_classes)
    y_test = kutils.to_categorical(y_test, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=koptimizers.RMSprop(),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
class MnistPatternExample_dense_linear(unittest.TestCase):
    def test(self):
        np.random.seed(234354346)

        model = mlp_2dense(
            (1, 28, 28),
            10,
            dense_units=1024,
            dropout_rate=0.25,
        )

        data = fetch_data()
        train_model(model, data, epochs=10)
        model.set_weights(model.get_weights())

        analyzer = innvestigate.create_analyzer(
            "pattern.net", model, pattern_type="linear"
        )
        analyzer.fit(data[0], batch_size=256, verbose=0)

        patterns = analyzer._patterns
        W = model.get_weights()[0]
        W2D = W.reshape((-1, W.shape[-1]))
        X = data[0].reshape((data[0].shape[0], -1))
        Y = np.dot(X, W2D)

        mean_x = X.mean(axis=0)
        mean_y = Y.mean(axis=0)
        mean_xy = np.dot(X.T, Y) / Y.shape[0]
        ExEy = mean_x[:, None] * mean_y[None, :]
        cov_xy = mean_xy - ExEy
        w_cov_xy = np.diag(np.dot(W2D.T, cov_xy))
        A = ibackend.safe_divide(cov_xy, w_cov_xy[None, :], factor=1)

        def allclose(a, b):
            return np.allclose(a, b, rtol=0.05, atol=0.05)

        # print(A.sum(), patterns[0].sum())
        self.assertTrue(allclose(A.ravel(), patterns[0].ravel()))


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
class MnistPatternExample_dense_relu(unittest.TestCase):
    def test(self):
        np.random.seed(234354346)

        model = mlp_2dense(
            (1, 28, 28),
            10,
            dense_units=1024,
            dropout_rate=0.25,
        )

        data = fetch_data()
        train_model(model, data, epochs=10)
        model.set_weights(model.get_weights())

        analyzer = innvestigate.create_analyzer(
            "pattern.net", model, pattern_type="relu"
        )
        analyzer.fit(data[0], batch_size=256, verbose=0)
        patterns = analyzer._patterns
        W, b = model.get_weights()[:2]
        W2D = W.reshape((-1, W.shape[-1]))
        X = data[0].reshape((data[0].shape[0], -1))
        Y = np.dot(X, W2D)

        mask = np.dot(X, W2D) + b > 0
        count = mask.sum(axis=0)

        mean_x = ibackend.safe_divide(np.dot(X.T, mask), count, factor=1)
        mean_y = Y.mean(axis=0)
        mean_xy = ibackend.safe_divide(np.dot(X.T, Y * mask), count, factor=1)

        ExEy = mean_x * mean_y

        cov_xy = mean_xy - ExEy
        w_cov_xy = np.diag(np.dot(W2D.T, cov_xy))
        A = ibackend.safe_divide(cov_xy, w_cov_xy[None, :], factor=1)

        def allclose(a, b):
            return np.allclose(a, b, rtol=0.05, atol=0.05)

        # print(A.sum(), patterns[0].sum())
        self.assertTrue(allclose(A.ravel(), patterns[0].ravel()))
