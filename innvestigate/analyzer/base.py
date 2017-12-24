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
from .. import utils as iutils
from ..utils.keras import graph

import keras.layers
import keras.models
import numpy as np


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
        tmp = self._create_analysis(model)
        try:
            analysis_output, debug_output = tmp
        except (TypeError, ValueError):
            analysis_output, debug_output = iutils.listify(tmp), list()

        self._n_debug_output = len(debug_output)
        self._analyzer_model = keras.models.Model(
            inputs=model_inputs+neuron_selection_inputs,
            outputs=analysis_output+debug_output)
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
            ret = self._analyzer_model.predict_on_batch(X, neuron_selection)
        else:
            ret = self._analyzer_model.predict_on_batch(X)

        if self._n_debug_output > 0:
            self._handle_debug_output(ret[-self._n_debug_output:])
            ret = ret[:-self._n_debug_output]

        if isinstance(ret, list) and len(ret) == 1:
            ret = ret[0]
        return ret


class BaseReverseNetwork(BaseNetwork):

    # Should be specified by the base class.
    reverse_mappings = {}
    default_reverse = None

    def __init__(self, *args,
                 reverse_verbose=False, reverse_check_finite=False, **kwargs):
        self._reverse_verbose = reverse_verbose
        self._reverse_check_finite = reverse_check_finite
        return super(BaseReverseNetwork, self).__init__(*args, **kwargs)

    def _create_analysis(self, model):
        ret = graph.reverse_model(
            model,
            reverse_mapping=self.reverse_mappings,
            default_reverse=self.default_reverse,
            verbose=self._reverse_verbose,
            return_all_reversed_tensors=self._reverse_check_finite)

        if self._reverse_check_finite is True:
            values = list(six.itervalues(ret[1]))
            mapping = {i: v["id"] for i, v in enumerate(values)}
            tensors = [v["tensor"] for v in values]
            ret = (ret[0], iutils.listify(ilayers.FiniteCheck()(tensors)))
            self._reverse_tensors_mapping = mapping

        return ret

    def _handle_debug_output(self, debug_values):
        nfinite_tensors = np.flatnonzero(np.asarray(debug_values) > 0)

        if len(nfinite_tensors) > 0:
            nfinite_tensors = [self._reverse_tensors_mapping[i]
                               for i in nfinite_tensors]
            print("Not finite values found in following nodes: "
                  "(ReverseID, TensorID) - {}".format(nfinite_tensors))
        pass
