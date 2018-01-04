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


import keras.layers
import keras.models
import keras.optimizers
import numpy as np
import unittest

# todo:fix relative imports:
#from ...utils.tests import dryrun

from innvestigate.utils.tests import dryrun

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
        n = 10000

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
        self.assertTrue(model.evaluate(X, y, verbose=0) < 0.01)

        pc = PatternComputer(model, pattern_type="linear")
        A = pc.compute(X)[0]
        W = model.get_weights()[0]

        #print(a_d, model.get_weights()[0])
        #print(a_s, A)

        def allclose(a, b):
            return np.allclose(a, b, rtol=0.01, atol=0.01)

        # perpendicular to a_d
        self.assertTrue(allclose(a_d.ravel(), abs(W.ravel())))
        # estimated pattern close to true pattern
        self.assertTrue(allclose(a_s.ravel(), A.ravel()))
