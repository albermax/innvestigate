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


import keras.backend as K
import keras.models
import numpy as np
import unittest

from . import networks


__all__ = [
    "BaseTestCase",
    "AnalyzerTestCase",
]


class BaseTestCase(unittest.TestCase):
    """
    A dryrun test on various networks for an analyzing method.

    For each network the test check that the generated network
    has the right output shape, can be compiled
    and executed with random inputs.
    """

    def _apply_test(self, method, network):
        raise NotImplementedError("Set in subclass.")

    def test_dryrun(self):
        # test shapes have channels first.
        # todo: check why import above fails
        import keras.backend as K 
        K.set_image_data_format("channels_first")

        for network in networks.iterator():
            if six.PY2:
                self._apply_test(self._method, network)
            else:
                with self.subTest(network_name=network["name"]):
                    self._apply_test(self._method, network)
        pass


class AnalyzerTestCase(BaseTestCase):

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _assert(self, method, network, x, explanation):
        pass

    def _apply_test(self, method, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        # Get analyzer.
        analyzer = method(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))
        analysis = analyzer.analyze(x)
        self.assertEqual(tuple(analysis.shape[1:]),
                         tuple(network["input_shape"][1:]))
        self._assert(method, network, x, analysis)
        pass


class PatternComputerTestCase(BaseTestCase):

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _assert(self, method, network, x, patterns):
        pass

    def _apply_test(self, method, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        # Get analyzer.
        analyzer = method(model)
        # Dryrun.
        x = np.random.rand(10, *(network["input_shape"][1:]))
        patterns = computer.compute_patterns(x, 2)
        self._assert(method, network, x, patterns)
        pass
