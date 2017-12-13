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


import numpy as np
import unittest

from . import networks


__all__ = [
    "BaseTestCase",
    "ExplainerTestCase",
]


class BaseTestCase(unittest.TestCase):
    """
    A dryrun test on various networks for an explanation method.

    For each network the test check that the generated network
    has the right output shape, can be compiled
    and executed with random inputs.
    """

    def _apply_test(self, method, network):
        raise NotImplementedError("Set in subclass.")

    def test_dryrun(self):
        for network in networks.iterator():
            if six.PY2:
                self._apply_test(self._method, network)
            else:
                with self.subTest(network_name=network["name"]):
                    self._apply_test(self._method, network)
        pass


class ExplainerTestCase(BaseTestCase):

    def _method(self, output_layer):
        raise NotImplementedError("Set in subclass.")

    def _assert(self, method, network, x, explanation):
        pass

    def _apply_test(self, method, network):
        # Get explainer.
        explainer = method(network["out"])
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))
        explanation = explainer.explain(x)
        self.assertEqual(tuple(explanation.shape[1:]),
                         tuple(network["input_shape"][1:]))
        self._assert(method, network, x, explanation)
        pass


class PatternComputerTestCase(BaseTestCase):

    def _method(self, output_layer):
        raise NotImplementedError("Set in subclass.")

    def _assert(self, method, network, x, explanation):
        pass

    def _apply_test(self, method, network):
        # Get explainer.
        computer = method(network["out"])
        # Dryrun.
        x = np.random.rand(10, *(network["input_shape"][1:]))
        patterns = computer.compute_patterns(x, 2)
        self._assert(method, network, x, patterns)
        pass
