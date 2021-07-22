from __future__ import annotations

import keras
import keras.backend as K
import keras.layers
import keras.models
import numpy as np

import innvestigate.layers as ilayers
import innvestigate.utils as iutils
import innvestigate.utils.keras.checks as kchecks
from innvestigate.analyzer.base import AnalyzerBase

__all__ = ["AnalyzerNetworkBase"]


class AnalyzerNetworkBase(AnalyzerBase):
    """Convenience interface for analyzers.

    This class provides helpful functionality to create analyzer's.
    Basically it:

    * takes the input model and adds a layer that selects
      the desired output neuron to analyze.
    * passes the new model to :func:`_create_analysis` which should
      return the analysis as Keras tensors.
    * compiles the function and serves the output to :func:`analyze` calls.
    * allows :func:`_create_analysis` to return tensors
      that are intercept for debugging purposes.

    :param neuron_selection_mode: How to select the neuron to analyze.
      Possible values are 'max_activation', 'index' for the neuron
      (expects indices at :func:`analyze` calls), 'all' take all neurons.
    :param allow_lambda_layers: Allow the model to contain lambda layers.
    """

    def __init__(
        self,
        model,
        neuron_selection_mode="max_activation",
        allow_lambda_layers=False,
        **kwargs
    ):
        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection parameter is not valid.")
        self._neuron_selection_mode = neuron_selection_mode

        self._allow_lambda_layers = allow_lambda_layers
        self._add_model_check(
            lambda layer: (
                not self._allow_lambda_layers
                and isinstance(layer, keras.layers.core.Lambda)
            ),
            (
                "Lamda layers are not allowed. "
                "To force use set allow_lambda_layers parameter."
            ),
            check_type="exception",
        )

        self._special_helper_layers = []

        super(AnalyzerNetworkBase, self).__init__(model, **kwargs)

    def _add_model_softmax_check(self):
        """
        Adds check that prevents models from containing a softmax.
        """
        self._add_model_check(
            lambda layer: kchecks.contains_activation(layer, activation="softmax"),
            "This analysis method does not support softmax layers.",
            check_type="exception",
        )

    def _prepare_model(self, model):
        """
        Prepares the model to analyze before it gets actually analyzed.

        This class adds the code to select a specific output neuron.
        """
        neuron_selection_mode = self._neuron_selection_mode
        model_inputs = model.inputs

        model_output = model.outputs
        if len(model_output) > 1:
            raise ValueError("Only models with one output tensor are allowed.")
        analysis_inputs = []
        stop_analysis_at_tensors = []

        # Flatten to form (batch_size, other_dimensions):
        if K.ndim(model_output[0]) > 2:
            model_output = keras.layers.Flatten()(model_output)

        if neuron_selection_mode == "max_activation":
            l = ilayers.Max(name="iNNvestigate_max")
            model_output = l(model_output)
            self._special_helper_layers.append(l)
        elif neuron_selection_mode == "index":
            neuron_indexing = keras.layers.Input(
                batch_shape=[None, None],
                dtype=np.int32,
                name="iNNvestigate_neuron_indexing",
            )
            self._special_helper_layers.append(neuron_indexing._keras_history[0])
            analysis_inputs.append(neuron_indexing)
            # The indexing tensor should not be analyzed.
            stop_analysis_at_tensors.append(neuron_indexing)

            l = ilayers.GatherND(name="iNNvestigate_gather_nd")
            model_output = l(model_output + [neuron_indexing])
            self._special_helper_layers.append(l)
        elif neuron_selection_mode == "all":
            pass
        else:
            raise NotImplementedError()

        model = keras.models.Model(
            inputs=model_inputs + analysis_inputs, outputs=model_output
        )
        return model, analysis_inputs, stop_analysis_at_tensors

    def create_analyzer_model(self):
        """
        Creates the analyze functionality. If not called beforehand
        it will be called by :func:`analyze`.
        """
        model_inputs = self._model.inputs
        tmp = self._prepare_model(self._model)
        model, analysis_inputs, stop_analysis_at_tensors = tmp
        self._analysis_inputs = analysis_inputs
        self._prepared_model = model

        tmp = self._create_analysis(
            model, stop_analysis_at_tensors=stop_analysis_at_tensors
        )
        if isinstance(tmp, tuple):
            if len(tmp) == 3:
                analysis_outputs, debug_outputs, constant_inputs = tmp
            elif len(tmp) == 2:
                analysis_outputs, debug_outputs = tmp
                constant_inputs = list()
            elif len(tmp) == 1:
                analysis_outputs = iutils.to_list(tmp[0])
                constant_inputs, debug_outputs = list(), list()
            else:
                raise Exception("Unexpected output from _create_analysis.")
        else:
            analysis_outputs = tmp
            constant_inputs, debug_outputs = list(), list()

        analysis_outputs = iutils.to_list(analysis_outputs)
        debug_outputs = iutils.to_list(debug_outputs)
        constant_inputs = iutils.to_list(constant_inputs)

        self._n_data_input = len(model_inputs)
        self._n_constant_input = len(constant_inputs)
        self._n_data_output = len(analysis_outputs)
        self._n_debug_output = len(debug_outputs)
        self._analyzer_model = keras.models.Model(
            inputs=model_inputs + analysis_inputs + constant_inputs,
            outputs=analysis_outputs + debug_outputs,
        )

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        """
        Interface that needs to be implemented by a derived class.

        This function is expected to create a Keras graph that creates
        a custom analysis for the model inputs given the model outputs.

        :param model: Target of analysis.
        :param stop_analysis_at_tensors: A list of tensors where to stop the
          analysis. Similar to stop_gradient arguments when computing the
          gradient of a graph.
        :return: Either one-, two- or three-tuple of lists of tensors.
          * The first list of tensors represents the analysis for each
            model input tensor. Tensors present in stop_analysis_at_tensors
            should be omitted.
          * The second list, if present, is a list of debug tensors that will
            be passed to :func:`_handle_debug_output` after the analysis
            is executed.
          * The third list, if present, is a list of constant input tensors
            added to the analysis model.
        """
        raise NotImplementedError()

    def _handle_debug_output(self, debug_values):
        raise NotImplementedError()

    def analyze(self, X, neuron_selection=None):
        """
        Same interface as :class:`Analyzer` besides

        :param neuron_selection: If neuron_selection_mode is 'index' this
          should be an integer with the index for the chosen neuron.
        """
        if not hasattr(self, "_analyzer_model"):
            self.create_analyzer_model()

        X = iutils.to_list(X)

        if neuron_selection is not None and self._neuron_selection_mode != "index":
            raise ValueError(
                "Only neuron_selection_mode 'index' expects "
                "the neuron_selection parameter."
            )
        if neuron_selection is None and self._neuron_selection_mode == "index":
            raise ValueError(
                "neuron_selection_mode 'index' expects "
                "the neuron_selection parameter."
            )

        if self._neuron_selection_mode == "index":
            neuron_selection = np.asarray(neuron_selection).flatten()
            if neuron_selection.size == 1:
                neuron_selection = np.repeat(neuron_selection, len(X[0]))

            # Add first axis indices for gather_nd
            neuron_selection = np.hstack(
                (
                    np.arange(len(neuron_selection)).reshape((-1, 1)),
                    neuron_selection.reshape((-1, 1)),
                )
            )
            ret = self._analyzer_model.predict_on_batch(X + [neuron_selection])
        else:
            ret = self._analyzer_model.predict_on_batch(X)

        if self._n_debug_output > 0:
            self._handle_debug_output(ret[-self._n_debug_output :])
            ret = ret[: -self._n_debug_output]

        if isinstance(ret, list) and len(ret) == 1:
            ret = ret[0]
        return ret

    def _get_state(self):
        state = super(AnalyzerNetworkBase, self)._get_state()
        state.update({"neuron_selection_mode": self._neuron_selection_mode})
        state.update({"allow_lambda_layers": self._allow_lambda_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        neuron_selection_mode = state.pop("neuron_selection_mode")
        allow_lambda_layers = state.pop("allow_lambda_layers")
        kwargs = super(AnalyzerNetworkBase, clazz)._state_to_kwargs(state)
        kwargs.update(
            {
                "neuron_selection_mode": neuron_selection_mode,
                "allow_lambda_layers": allow_lambda_layers,
            }
        )
        return kwargs
