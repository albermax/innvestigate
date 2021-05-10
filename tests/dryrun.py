# Get Python six functionality:
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import keras.backend as K
import keras.models
import numpy as np
import six

from innvestigate.analyzer.base import AnalyzerBase

from tests import networks

###############################################################################
###############################################################################
###############################################################################


__all__ = [
    "AnalyzerTestCase",
    "EqualAnalyzerTestCase",
    "PatternComputerTestCase",
]


###############################################################################
###############################################################################
###############################################################################


def _set_zero_weights_to_random(weights):
    ret = []
    for weight in weights:
        if weight.sum() == 0:
            weight = np.random.rand(*weight.shape)
        ret.append(weight)
    return ret


###############################################################################
###############################################################################
###############################################################################


class BaseLayerTestCase(unittest.TestCase):
    """
    A dryrun test on various networks for an analyzing method.

    For each network the test check that the generated network
    has the right output shape, can be compiled
    and executed with random inputs.
    """

    _network_filter = "trivia.*"

    def __init__(self, *args, **kwargs):
        network_filter = kwargs.pop("network_filter", None)
        if network_filter is not None:
            self._network_filter = network_filter
        super(BaseLayerTestCase, self).__init__(*args, **kwargs)

    def _apply_test(self, network):
        raise NotImplementedError("Set in subclass.")

    def runTest(self):
        np.random.seed(2349784365)
        K.clear_session()

        for network in networks.iterator(self._network_filter, clear_sessions=True):
            if six.PY2:
                self._apply_test(network)
            else:
                with self.subTest(network_name=network["name"]):
                    self._apply_test(network)


###############################################################################
###############################################################################
###############################################################################


class AnalyzerTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies the method to several networks and
    runs the analyzer with random data.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, *args, **kwargs):
        method = kwargs.pop("method", None)
        if method is not None:
            self._method = method
        super(AnalyzerTestCase, self).__init__(*args, **kwargs)

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))
        analysis = analyzer.analyze(x)
        self.assertEqual(
            tuple(analysis.shape), (1,) + tuple(network["input_shape"][1:])
        )
        self.assertFalse(np.any(np.isinf(analysis.ravel())))
        self.assertFalse(np.any(np.isnan(analysis.ravel())))


def test_analyzer(method, network_filter):
    """Workaround for move from unit-tests to pytest."""
    # todo: Mixing of pytest and unittest is not ideal.
    # Move completely to pytest.
    test_case = AnalyzerTestCase(method=method, network_filter=network_filter)
    test_result = unittest.TextTestRunner().run(test_case)
    assert len(test_result.errors) == 0
    assert len(test_result.failures) == 0


class AnalyzerTrainTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies the method to several networks and
    trains and runs the analyzer with random data.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, *args, **kwargs):
        method = kwargs.pop("method", None)
        if method is not None:
            self._method = method
        super(AnalyzerTrainTestCase, self).__init__(*args, **kwargs)

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(16, *(network["input_shape"][1:]))
        analyzer.fit(x)
        x = np.random.rand(1, *(network["input_shape"][1:]))
        analysis = analyzer.analyze(x)
        self.assertEqual(
            tuple(analysis.shape), (1,) + tuple(network["input_shape"][1:])
        )
        self.assertFalse(np.any(np.isinf(analysis.ravel())))
        self.assertFalse(np.any(np.isnan(analysis.ravel())))
        self.assertFalse(True)


def test_train_analyzer(method, network_filter):
    """Workaround for move from unit-tests to pytest."""
    # todo: Mixing of pytest and unittest is not ideal.
    # Move completely to pytest.
    test_case = AnalyzerTrainTestCase(method=method, network_filter=network_filter)
    test_result = unittest.TextTestRunner().run(test_case)
    assert len(test_result.errors) == 0
    assert len(test_result.failures) == 0


class EqualAnalyzerTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies two method to several networks and
    runs the analyzer with random data and checks for equality
    of the results.

    :param method1: A function that returns an Analyzer class.
    :param method2: A function that returns an Analyzer class.
    """

    def __init__(self, *args, **kwargs):
        method1 = kwargs.pop("method1", None)
        method2 = kwargs.pop("method2", None)
        if method1 is not None:
            self._method1 = method1
        if method2 is not None:
            self._method2 = method2

        super(EqualAnalyzerTestCase, self).__init__(*args, **kwargs)

    def _method1(self, model):
        raise NotImplementedError("Set in subclass.")

    def _method2(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer1 = self._method1(model)
        analyzer2 = self._method2(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:])) * 100
        analysis1 = analyzer1.analyze(x)
        analysis2 = analyzer2.analyze(x)

        self.assertEqual(
            tuple(analysis1.shape), (1,) + tuple(network["input_shape"][1:])
        )
        self.assertFalse(np.any(np.isinf(analysis1.ravel())))
        self.assertFalse(np.any(np.isnan(analysis1.ravel())))
        self.assertEqual(
            tuple(analysis2.shape), (1,) + tuple(network["input_shape"][1:])
        )
        self.assertFalse(np.any(np.isinf(analysis2.ravel())))
        self.assertFalse(np.any(np.isnan(analysis2.ravel())))

        all_close_kwargs = {}
        if hasattr(self, "_all_close_rtol"):
            all_close_kwargs["rtol"] = self._all_close_rtol
        if hasattr(self, "_all_close_atol"):
            all_close_kwargs["atol"] = self._all_close_atol
        # print(analysis1.sum(), analysis2.sum())
        self.assertTrue(np.allclose(analysis1, analysis2, **all_close_kwargs))


def test_equal_analyzer(method1, method2, network_filter):
    """Workaround for move from unit-tests to pytest."""
    # todo: Mixing of pytest and unittest is not ideal.
    # Move completely to pytest.
    test_case = EqualAnalyzerTestCase(
        method1=method1, method2=method2, network_filter=network_filter
    )
    test_result = unittest.TextTestRunner().run(test_case)
    assert len(test_result.errors) == 0
    assert len(test_result.failures) == 0


# todo: merge with base test case? if we don't run the analysis
# its only half the test.
class SerializeAnalyzerTestCase(BaseLayerTestCase):
    """TestCase for analyzers serialization

    TestCase that applies the method to several networks and
    runs the analyzer with random data, serializes it, and
    runs it again.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, *args, **kwargs):
        method = kwargs.pop("method", None)
        if method is not None:
            self._method = method
        super(SerializeAnalyzerTestCase, self).__init__(*args, **kwargs)

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))

        class_name, state = analyzer.save()
        new_analyzer = AnalyzerBase.load(class_name, state)

        analysis = new_analyzer.analyze(x)
        self.assertEqual(
            tuple(analysis.shape), (1,) + tuple(network["input_shape"][1:])
        )
        self.assertFalse(np.any(np.isinf(analysis.ravel())))
        self.assertFalse(np.any(np.isnan(analysis.ravel())))


def test_serialize_analyzer(method, network_filter):
    """Workaround for move from unit-tests to pytest."""
    # todo: Mixing of pytest and unittest is not ideal.
    # Move completely to pytest.
    test_case = SerializeAnalyzerTestCase(method=method, network_filter=network_filter)
    test_result = unittest.TextTestRunner().run(test_case)
    assert len(test_result.errors) == 0
    assert len(test_result.failures) == 0


###############################################################################
###############################################################################
###############################################################################


class PatternComputerTestCase(BaseLayerTestCase):
    """TestCase pattern computation

    :param method: A function that returns an PatternComputer class.
    """

    def __init__(self, *args, **kwargs):
        method = kwargs.pop("method", None)
        if method is not None:
            self._method = method
        super(PatternComputerTestCase, self).__init__(*args, **kwargs)

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = keras.models.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get computer.
        computer = self._method(model)
        # Dryrun.
        x = np.random.rand(10, *(network["input_shape"][1:]))
        computer.compute(x)


def test_pattern_computer(method, network_filter):
    """Workaround for move from unit-tests to pytest."""
    # todo: Mixing of pytest and unittest is not ideal.
    # Move completely to pytest.
    test_case = PatternComputerTestCase(method=method, network_filter=network_filter)
    test_result = unittest.TextTestRunner().run(test_case)
    assert len(test_result.errors) == 0
    assert len(test_result.failures) == 0
