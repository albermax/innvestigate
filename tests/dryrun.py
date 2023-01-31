"""Dryrun analyzers to catch crashes."""
from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow
import tensorflow.keras.backend as kbackend

tensorflow.compat.v1.disable_eager_execution()

from innvestigate.analyzer import Random
from innvestigate.analyzer.base import AnalyzerBase
from innvestigate.backend.types import Model

from tests import networks


class BaseLayerTestCase:
    """
    A dryrun test on various networks for an analyzing method.

    For each network the test check that the generated network
    has the right output shape, can be compiled
    and executed with random inputs.
    """

    def __init__(self, network_filter: str = None) -> None:
        self._network_filter = network_filter

    def _apply_test(self, _model: Model) -> None:
        raise NotImplementedError("Set in subclass.")

    def run_test(self) -> None:
        np.random.seed(2349784365)
        kbackend.clear_session()

        for model in networks.iterator(self._network_filter, clear_sessions=True):
            self._apply_test(model)


###############################################################################


class AnalyzerTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies the method to several networks and
    runs the analyzer with random data.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, method: AnalyzerBase, *args, **kwargs) -> None:
        self._method = method
        super().__init__(*args, **kwargs)

    def _apply_test(self, model: Model) -> None:
        # Generate random test input
        input_shape = model.input_shape[1:]
        x = np.random.rand(1, *input_shape).astype(np.float32)
        # Call model with test input
        model.predict(x)
        # Call analyzer
        analyzer = self._method(model)
        analysis = analyzer.analyze(x)

        assert tuple(analysis.shape) == (1,) + input_shape
        assert not np.any(np.isinf(analysis.ravel()))
        assert not np.any(np.isnan(analysis.ravel()))


def test_analyzer(method: Callable, network_filter: str) -> None:
    """Workaround for move from unit-tests to pytest."""
    test_case = AnalyzerTestCase(method, network_filter=network_filter)
    test_case.run_test()


class AnalyzerTrainTestCase(BaseLayerTestCase):
    """TestCase for analyzers execution

    TestCase that applies the method to several networks and
    trains and runs the analyzer with random data.

    :param method: A function that returns an Analyzer class.
    """

    def __init__(self, *args, method=None, **kwargs) -> None:
        if method is not None:
            self._method = method
        super().__init__(*args, **kwargs)

    def _method(self, _model: Model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, model: Model) -> None:
        # Generate random training input
        input_shape = model.input_shape[1:]
        x = np.random.rand(1, *input_shape).astype(np.float32)
        # Call model with test input
        model.predict(x)
        # Get analyzer.
        analyzer = self._method(model)
        # Generate random test input
        analysis = analyzer.analyze(x)
        assert tuple(analysis.shape) == (1,) + input_shape
        assert not np.any(np.isinf(analysis.ravel()))
        assert not np.any(np.isnan(analysis.ravel()))


def test_train_analyzer(method, network_filter) -> None:
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

    def __init__(self, *args, method1=None, method2=None, **kwargs) -> None:
        if method1 is not None:
            self._method1 = method1
        if method2 is not None:
            self._method2 = method2

        super().__init__(*args, **kwargs)

    def _method1(self, _model: Model):
        raise NotImplementedError("Set in subclass.")

    def _method2(self, _model: Model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, model: Model) -> None:
        # Generate random training input
        input_shape = model.input_shape[1:]
        x = np.random.rand(1, *input_shape).astype(np.float32)
        # Call model with test input
        model.predict(x)
        # Get analyzer.
        analyzer1 = self._method1(model)
        analyzer2 = self._method2(model)
        # Dryrun.
        analysis1 = analyzer1.analyze(x)
        analysis2 = analyzer2.analyze(x)

        assert tuple(analysis1.shape) == (1,) + input_shape
        assert not np.any(np.isinf(analysis1.ravel()))
        assert not np.any(np.isnan(analysis1.ravel()))

        assert tuple(analysis2.shape) == (1,) + input_shape
        assert not np.any(np.isinf(analysis2.ravel()))
        assert not np.any(np.isnan(analysis2.ravel()))
        assert np.allclose(analysis1, analysis2)


def test_equal_analyzer(method1, method2, network_filter) -> None:
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

    def __init__(self, *args, method=None, **kwargs) -> None:
        if method is not None:
            self._method = method
        super().__init__(*args, **kwargs)

    def _method(self, model: Model):
        raise NotImplementedError("Set in subclass.")

    def _apply_test(self, model: Model) -> None:
        # Get analyzer.
        analyzer = self._method(model)
        # Dryrun.
        input_shape = model.input_shape[1:]
        x = np.random.rand(1, *input_shape)
        old_analysis = analyzer.analyze(x)

        class_name, state = analyzer.save()
        new_analyzer = AnalyzerBase.load(class_name, state)

        new_analysis = new_analyzer.analyze(x)
        assert tuple(new_analysis.shape) == (1,) + input_shape
        assert not np.any(np.isinf(new_analysis.ravel()))
        assert not np.any(np.isnan(new_analysis.ravel()))

        # Check equality of analysis for all deterministic analyzers
        if not isinstance(analyzer, (Random)):
            assert np.allclose(new_analysis, old_analysis)


def test_serialize_analyzer(method, network_filter) -> None:
    """Workaround for move from unit-tests to pytest."""
    test_case = SerializeAnalyzerTestCase(method=method, network_filter=network_filter)
    test_case.run_test()
