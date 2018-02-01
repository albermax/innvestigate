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


import keras.backend as K
import keras.models
import numpy as np
import unittest

from ...analyzer import BaseAnalyzer
from . import networks


__all__ = [
    "BaseTestCase",
    "AnalyzerTestCase",
    "EqualAnalyzerTestCase",
    "PatternComputerTestCase",
]


###############################################################################
###############################################################################
###############################################################################


class BaseTestCase(unittest.TestCase):
    """
    A dryrun test on various networks for an analyzing method.

    For each network the test check that the generated network
    has the right output shape, can be compiled
    and executed with random inputs.
    """

    def _apply_test(self, network):
        raise NotImplementedError("Set in subclass.")

    def test_dryrun(self):
        # test shapes have channels first.
        # todo: check why import above fails
        import keras.backend as K 
        K.set_image_data_format("channels_first")

        for network in networks.iterator():
            if six.PY2:
                self._apply_test(network)
            else:
                with self.subTest(network_name=network["name"]):
                    self._apply_test(network)
        pass


###############################################################################
###############################################################################
###############################################################################


class AnalyzerTestCase(BaseTestCase):

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))
        analysis = analyzer.analyze(x)
        self.assertEqual(tuple(analysis.shape[1:]),
                         tuple(network["input_shape"][1:]))
        self.assertFalse(np.any(np.isnan(analysis.ravel())))
        pass


class EqualAnalyzerTestCase(BaseTestCase):

    def _method1(self, model):
        raise NotImplementedError("Set in subclass.")

    def _method2(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        # Get analyzer.
        analyzer1 = self._method1(model)
        analyzer2 = self._method2(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))
        analysis1 = analyzer1.analyze(x)
        analysis2 = analyzer2.analyze(x)

        self.assertEqual(tuple(analysis1.shape[1:]),
                         tuple(network["input_shape"][1:]))
        self.assertFalse(np.any(np.isnan(analysis1.ravel())))
        self.assertEqual(tuple(analysis2.shape[1:]),
                         tuple(network["input_shape"][1:]))
        self.assertFalse(np.any(np.isnan(analysis2.ravel())))
        self.assertTrue(np.allclose(analysis1, analysis2))
        pass


# todo: merge with base test case? if we don't run the analysis
# its only half the test.
class SerializeAnalyzerTestCase(BaseTestCase):

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))

        class_name, state = analyzer.save()
        new_analyzer = BaseAnalyzer.load(class_name, state)

        analysis = new_analyzer.analyze(x)
        self.assertEqual(tuple(analysis.shape[1:]),
                         tuple(network["input_shape"][1:]))
        self.assertFalse(np.any(np.isnan(analysis.ravel())))
        pass


###############################################################################
###############################################################################
###############################################################################


class PatternComputerTestCase(BaseTestCase):

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        # Get computer.
        computer = self._method(model)
        # Dryrun.
        x = np.random.rand(10, *(network["input_shape"][1:]))
        patterns = computer.compute(x)
        pass
