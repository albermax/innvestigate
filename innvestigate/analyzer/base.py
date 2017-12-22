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


from .. import layers as ilayers
from ..utils.keras import graph

import keras.layers
import keras.models


__all__ = [
    "Base",
    "BaseNetwork",
    "BaseReverseNetwork"
]


###############################################################################
###############################################################################
###############################################################################


class Base(object):

    properties = {
        "name": "undefined",
        "show_as": "undefined",
    }

    def __init__(self, model):
        self._model = model
        pass

    def explain(self, X):
        raise NotImplementedError("Has to be implemented by the subclass")


###############################################################################
###############################################################################
###############################################################################


class BaseNetwork(Base):

    def __init__(self, model, neuron_selection_mode="max_activation"):
        super(BaseNetwork, self).__init__(model)

        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection parameter is not valid.")
        self._neuron_selection_mode = neuron_selection_mode

        neuron_selection_inputs = []
        model_inputs, model_output = model.inputs, model.outputs
        if len(model_output) > 1:
            raise ValueError("Only models with one output tensor are allowed.")

        if neuron_selection_mode == "max_activation":
            model_output = ilayers.Max()(model_output)
        if neuron_selection_mode == "index":
            raise NotImplementedError("Only a stub present so far.")
            neuron_indexing = keras.layers.Input(shape=[None, None])
            neuron_selection_inputs += neuron_indexing

            model_output = keras.layers.Index()([model_output, neuron_indexing])

        model = keras.models.Model(inputs=model_inputs+neuron_selection_inputs,
                                   outputs=model_output)
        analysis_output = self._create_analysis(model)

        self._analyzer_model = keras.models.Model(
            inputs=model_inputs+neuron_selection_inputs,
            outputs=analysis_output)
        self._analyzer_model.compile(optimizer="sgd", loss="mse")
        pass

    def _create_analysis(self, model):
        raise NotImplementedError()

    def analyze(self, X, neuron_selection=None):
        if(neuron_selection is not None and
           self._neuron_selection_mode != "index"):
            raise ValueError("Only neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")
        if(neuron_selection is None and
           self._neuron_selection_mode == "index"):
            raise ValueError("neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")

        if self._neuron_selection_mode == "index":
            return self._analyzer_model.predict_on_batch(X, neuron_selection)
        else:
            return self._analyzer_model.predict_on_batch(X)


class BaseReverseNetwork(BaseNetwork):

    # Should be specified by the base class.
    reverse_mappings = {}
    default_reverse = None

    def _create_analysis(self, model):
        return graph.reverse_model(model,
                                   reverse_mapping=self.reverse_mappings,
                                   default_reverse=self.default_reverse)
