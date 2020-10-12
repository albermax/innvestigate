# Get Python six functionality:
from __future__ import \
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate import backend
from innvestigate.utils.tests import cases
from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import BaselineGradient
from innvestigate.analyzer import Gradient


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__BasicGraphReversal(case_id):

    def create_analyzer1_f(model):
        return BaselineGradient(model)

    def create_analyzer2_f(model):
        return Gradient(model)

    dryrun.test_analyzers_for_same_output(
        case_id, create_analyzer1_f, create_analyzer2_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__BasicGraphReversal(case_id):

    def create_analyzer1_f(model):
        return BaselineGradient(model)

    def create_analyzer2_f(model):
        return Gradient(model)

    dryrun.test_analyzers_for_same_output(
        case_id, create_analyzer1_f, create_analyzer2_f)


@pytest.mark.skipif(backend.name() != "tensorflow",
                    reason="TensorFlow-specific test.")
# todo(alber): Enable for new TF backend.
@pytest.mark.xfail(reason="Missing/buggy feature.")
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__ContainerGraphReversal(case_id):

    def create_analyzer1_f(model):
        return Gradient(model)

    def create_analyzer2_f(model):
        # Create container execution
        model = backend.keras.models.Model(inputs=model.inputs,
                                           outputs=model(model.inputs))
        return Gradient(model)

    dryrun.test_analyzers_for_same_output(
        case_id, create_analyzer1_f, create_analyzer2_f)


@pytest.mark.skipif(backend.name() != "tensorflow",
                    reason="TensorFlow-specific test.")
# todo(alber): Enable for new TF backend.
@pytest.mark.xfail(reason="Missing/buggy feature.")
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__ContainerGraphReversal(case_id):

    def create_analyzer1_f(model):
        return Gradient(model)

    def create_analyzer2_f(model):
        # Create container execution
        model = backend.keras.models.Model(inputs=model.inputs,
                                           outputs=model(model.inputs))
        return Gradient(model)

    dryrun.test_analyzers_for_same_output(
        case_id, create_analyzer1_f, create_analyzer2_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__AnalyzerNetworkBase_neuron_selection_max(case_id):

    def create_analyzer_f(model):
        return Gradient(model, neuron_selection_mode="max_activation")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__AnalyzerNetworkBase_neuron_selection_max(case_id):

    def create_analyzer_f(model):
        return Gradient(model, neuron_selection_mode="max_activation")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__AnalyzerNetworkBase_neuron_selection_index(case_id):

    class CustomAnalyzer(Gradient):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def create_analyzer_f(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__AnalyzerNetworkBase_neuron_selection_index(case_id):

    class CustomAnalyzer(Gradient):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def create_analyzer_f(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(case_id, create_analyzer_f)
