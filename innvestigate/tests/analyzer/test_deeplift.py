# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import keras.layers
import keras.models
import numpy as np
import pytest
try:
    import deeplift
except ImportError:
    deeplift = None

from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import DeepLIFT
from innvestigate.analyzer import DeepLIFTWrapper


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DeepLIFT():

    def method(model):
        return DeepLIFT(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__DeepLIFT():

    def method(model):
        return DeepLIFT(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.precommit
def test_precommit__DeepLIFT_Rescale():

    def method(model):
        if keras.backend.image_data_format() == "channels_first":
            input_shape = (1, 28, 28)
        else:
            input_shape = (28, 28, 1)
        model = keras.models.Sequential([
            keras.layers.Dense(10, input_shape=input_shape),
            keras.layers.ReLU(),
        ])
        return DeepLIFT(model)

    dryrun.test_analyzer(method, "mnist.log_reg")


@pytest.mark.precommit
def test_precommit__DeepLIFT_neuron_selection_index():

    class CustomAnalyzer(DeepLIFT):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.precommit
def test_precommit__DeepLIFT_larger_batch_size():

    class CustomAnalyzer(DeepLIFT):

        def analyze(self, X):
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X)[0:1]

    def method(model):
        return CustomAnalyzer(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.skip("There is a design issue to be fixed.")
@pytest.mark.precommit
def test_precommit__DeepLIFT_larger_batch_size_with_index():

    class CustomAnalyzer(DeepLIFT):

        def analyze(self, X):
            index = 0
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X, index)[0:1]

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__DeepLIFT():

    def method(model):
        return DeepLIFT(model)

    dryrun.test_analyzer(method, "imagenet.*")


###############################################################################
###############################################################################
###############################################################################


require_deeplift = pytest.mark.skipif(deeplift is None,
                                      reason="Package deeplift is required.")


@require_deeplift
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.skip(reason="DeepLIFT does not work with skip connection.")
def test_fast__DeepLIFTWrapper():

    def method(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper():

    def method(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper_neuron_selection_index():

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper_larger_batch_size():

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X)[0:1]

    def method(model):
        return CustomAnalyzer(model)

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper_larger_batch_size_with_index():

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            index = 0
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X, index)[0:1]

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__DeepLIFTWrapper():

    def method(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(method, "imagenet.*")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DeepLIFT_serialize():

    def method(model):
        return DeepLIFT(model)

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DeepLIFTWrapper_serialize():

    def method(model):
        return DeepLIFTWrapper(model)

    with pytest.raises(AssertionError):
        # Issue in deeplift.
        dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")
