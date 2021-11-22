from __future__ import annotations

import numpy as np
import tensorflow.keras.backend as kbackend
import tensorflow.keras.models as kmodels

from innvestigate.analyzer.base import AnalyzerBase
from innvestigate.utils.types import Model

from tests import networks


def _set_zero_weights_to_random(weights: np.ndarray):
    ret = []
    for weight in weights:
        if weight.sum() == 0:
            weight = np.random.rand(*weight.shape)
        ret.append(weight)
    return ret


class BaseLayerTestCase:
    """
    A dryrun test on various networks for an analyzing method.

    For each network the test check that the generated network
    has the right output shape, can be compiled
    and executed with random inputs.
    """

    def __init__(self, network_filter: str = None) -> None:
        self._network_filter = network_filter

    def _apply_test(self, network: Model):
        raise NotImplementedError("Set in subclass.")

    def run_test(self):
        np.random.seed(2349784365)
        kbackend.clear_session()

        for network in networks.iterator(self._network_filter, clear_sessions=True):
            self._apply_test(network)


###############################################################################


class AnalyzerTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies the method to several networks and
    runs the analyzer with random data.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, method: AnalyzerBase, *args, **kwargs):
        self._method = method
        super().__init__(*args, **kwargs)

    def _apply_test(self, network: Model):
        # Create model.
        model = kmodels.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))
        analysis = analyzer.analyze(x)

        assert tuple(analysis.shape) == (1,) + tuple(network["input_shape"][1:])
        assert not np.any(np.isinf(analysis.ravel()))
        assert not np.any(np.isnan(analysis.ravel()))


def test_analyzer(method: AnalyzerBase, network_filter):
    """Workaround for move from unit-tests to pytest."""
    test_case = AnalyzerTestCase(method, network_filter=network_filter)
    test_case.run_test()


class AnalyzerTrainTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies the method to several networks and
    trains and runs the analyzer with random data.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, *args, method=None, **kwargs):
        if method is not None:
            self._method = method
        super().__init__(*args, **kwargs)

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = kmodels.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(16, *(network["input_shape"][1:]))
        analyzer.fit(x)
        x = np.random.rand(1, *(network["input_shape"][1:]))
        analysis = analyzer.analyze(x)
        assert tuple(analysis.shape) == (1,) + tuple(network["input_shape"][1:])
        assert not np.any(np.isinf(analysis.ravel()))
        assert not np.any(np.isnan(analysis.ravel()))


def test_train_analyzer(method, network_filter):
    """Workaround for move from unit-tests to pytest."""
    test_case = AnalyzerTrainTestCase(method=method, network_filter=network_filter)
    test_case.run_test()


class EqualAnalyzerTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies two method to several networks and
    runs the analyzer with random data and checks for equality
    of the results.

    :param method1: A function that returns an Analyzer class.
    :param method2: A function that returns an Analyzer class.
    """

    def __init__(self, *args, method1=None, method2=None, **kwargs):
        if method1 is not None:
            self._method1 = method1
        if method2 is not None:
            self._method2 = method2

        super().__init__(*args, **kwargs)

    def _method1(self, model):
        raise NotImplementedError("Set in subclass.")

    def _method2(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = kmodels.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer1 = self._method1(model)
        analyzer2 = self._method2(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:])) * 100
        analysis1 = analyzer1.analyze(x)
        analysis2 = analyzer2.analyze(x)

        assert tuple(analysis1.shape) == (1,) + tuple(network["input_shape"][1:])
        assert not np.any(np.isinf(analysis1.ravel()))
        assert not np.any(np.isnan(analysis1.ravel()))

        assert tuple(analysis2.shape) == (1,) + tuple(network["input_shape"][1:])
        assert not np.any(np.isinf(analysis2.ravel()))
        assert not np.any(np.isnan(analysis2.ravel()))

        all_close_kwargs = {}
        if hasattr(self, "_all_close_rtol"):
            all_close_kwargs["rtol"] = self._all_close_rtol
        if hasattr(self, "_all_close_atol"):
            all_close_kwargs["atol"] = self._all_close_atol
        # print(analysis1.sum(), analysis2.sum())
        assert np.allclose(analysis1, analysis2, **all_close_kwargs)


def test_equal_analyzer(method1, method2, network_filter):
    """Workaround for move from unit-tests to pytest."""
    test_case = EqualAnalyzerTestCase(
        method1=method1, method2=method2, network_filter=network_filter
    )
    test_case.run_test()


# todo: merge with base test case? if we don't run the analysis
# its only half the test.
class SerializeAnalyzerTestCase(BaseLayerTestCase):
    """TestCase for analyzers serialization

    TestCase that applies the method to several networks and
    runs the analyzer with random data, serializes it, and
    runs it again.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, *args, method=None, **kwargs):
        if method is not None:
            self._method = method
        super().__init__(*args, **kwargs)

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = kmodels.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        x = np.random.rand(1, *(network["input_shape"][1:]))

        class_name, state = analyzer.save()
        new_analyzer = AnalyzerBase.load(class_name, state)

        analysis = new_analyzer.analyze(x)
        assert tuple(analysis.shape) == (1,) + tuple(network["input_shape"][1:])
        assert not np.any(np.isinf(analysis.ravel()))
        assert not np.any(np.isnan(analysis.ravel()))


def test_serialize_analyzer(method, network_filter):
    """Workaround for move from unit-tests to pytest."""
    test_case = SerializeAnalyzerTestCase(method=method, network_filter=network_filter)
    test_case.run_test()


###############################################################################


class PatternComputerTestCase(BaseLayerTestCase):
    """TestCase pattern computation

    :param method: A function that returns an PatternComputer class.
    """

    def __init__(self, *args, method=None, **kwargs):
        if method is not None:
            self._method = method
        super().__init__(*args, **kwargs)

    def _method(self, model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, network):
        # Create model.
        model = kmodels.Model(inputs=network["in"], outputs=network["out"])
        model.set_weights(_set_zero_weights_to_random(model.get_weights()))
        # Get computer.
        computer = self._method(model)
        # Dryrun.
        x = np.random.rand(10, *(network["input_shape"][1:]))
        computer.compute(x)


def test_pattern_computer(method, network_filter):
    """Workaround for move from unit-tests to pytest."""
    test_case = PatternComputerTestCase(method=method, network_filter=network_filter)
    test_case.run_test()
