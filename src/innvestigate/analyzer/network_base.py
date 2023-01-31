from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.backend as ibackend
import innvestigate.backend.checks as ichecks
import innvestigate.layers as ilayers
from innvestigate.analyzer.base import AnalyzerBase
from innvestigate.backend.types import Layer, LayerCheck, Model, OptionalList, Tensor

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

    :param allow_lambda_layers: Allow the model to contain lambda layers.
    """

    def __init__(
        self,
        model: Model,
        allow_lambda_layers: bool = False,
        **kwargs,
    ) -> None:
        """
        From AnalyzerBase super init:
        * Initializes empty list of _model_checks
        * set _neuron_selection_mode

        Here:
        * add check for lambda layers through 'allow_lambda_layers'
        * define attributes for '_prepare_model', which is later called
            through 'create_analyzer_model'
        """
        # Call super init to initialize self._model_checks
        super().__init__(model, **kwargs)

        # Add model check for lambda layers
        self._allow_lambda_layers: bool = allow_lambda_layers
        self._add_lambda_layers_check()

        # Attributes of prepared model created by '_prepare_model'
        self._analyzer_model_done: bool = False
        self._analyzer_model: Model = None
        self._special_helper_layers: list[Layer] = []  # added for _reverse_mapping
        self._analysis_inputs: list[Tensor] | None = None
        self._n_data_input: int = 0
        self._n_constant_input: int = 0
        self._n_data_output: int = 0
        self._n_debug_output: int = 0

    def _add_model_softmax_check(self) -> None:
        """
        Adds check that prevents models from containing a softmax.
        """
        contains_softmax: LayerCheck = lambda layer: ichecks.contains_activation(
            layer, activation="softmax"
        )
        self._add_model_check(
            check=contains_softmax,
            message="This analysis method does not support softmax layers.",
            check_type="exception",
        )

    def _add_lambda_layers_check(self) -> None:
        check_lambda_layers: LayerCheck = lambda layer: (
            not self._allow_lambda_layers and isinstance(layer, klayers.Lambda)
        )
        self._add_model_check(
            check=check_lambda_layers,
            message=(
                "Lambda layers are not allowed. "
                "To force use set 'allow_lambda_layers' parameter."
            ),
            check_type="exception",
        )

    def _prepare_model(self, model: Model) -> tuple[Model, list[Tensor], list[Tensor]]:
        """
        Prepares the model to analyze before it gets actually analyzed.

        This class adds the code to select a specific output neuron.
        """
        neuron_selection_mode: str
        model_inputs: list[Tensor]
        model_output: list[Tensor]
        analysis_inputs: list[Tensor]
        stop_analysis_at_tensors: list[Tensor]

        neuron_selection_mode = self._neuron_selection_mode
        model_inputs = model.inputs
        model_output = model.outputs

        if len(model_output) > 1:
            raise ValueError("Only models with one output tensor are allowed.")

        analysis_inputs = []
        stop_analysis_at_tensors = []

        # Flatten to form (batch_size, other_dimensions):
        if kbackend.ndim(model_output[0]) > 2:
            model_output = klayers.Flatten()(model_output)

        if neuron_selection_mode == "max_activation":
            inn_max = ilayers.MaxNeuronSelection(name="MaxNeuronSelection")
            model_output = inn_max(model_output)
            self._special_helper_layers.append(inn_max)

        elif neuron_selection_mode == "index":
            # Creates a placeholder tensor when `dtype` is passed.
            neuron_indexing: Layer = klayers.Input(
                shape=(2,),  # infer amount of output neurons
                dtype=np.int32,
                name="iNNvestigate_neuron_indexing",
            )
            # TODO: what does _keras_history[0] do?
            self._special_helper_layers.append(neuron_indexing._keras_history[0])
            analysis_inputs.append(neuron_indexing)
            stop_analysis_at_tensors.append(neuron_indexing)

            select = ilayers.NeuronSelection(name="NeuronSelection")
            model_output = select(model_output + [neuron_indexing])
            self._special_helper_layers.append(select)
        elif neuron_selection_mode == "all":
            pass
        else:
            raise NotImplementedError()

        inputs = model_inputs + analysis_inputs
        outputs = model_output
        model = kmodels.Model(inputs=inputs, outputs=outputs)

        return model, analysis_inputs, stop_analysis_at_tensors

    def create_analyzer_model(self) -> None:
        """
        Creates the analyze functionality. If not called beforehand
        it will be called by :func:`analyze`.
        """
        model_inputs: list[Tensor] = self._model.inputs
        model, analysis_inputs, stop_analysis_at_tensors = self._prepare_model(
            self._model
        )
        self._analysis_inputs = analysis_inputs
        self._prepared_model = model

        tmp = self._create_analysis(
            model, stop_analysis_at_tensors=stop_analysis_at_tensors
        )
        if isinstance(tmp, tuple):
            if len(tmp) == 3:
                analysis_outputs, debug_outputs, constant_inputs = tmp  # type: ignore
            elif len(tmp) == 2:
                analysis_outputs, debug_outputs = tmp  # type: ignore
                constant_inputs = []
            elif len(tmp) == 1:
                analysis_outputs = tmp[0]
                constant_inputs = []
                debug_outputs = []
            else:
                raise Exception("Unexpected output from _create_analysis.")
        else:
            analysis_outputs = tmp
            constant_inputs = []
            debug_outputs = []

        analysis_outputs = ibackend.to_list(analysis_outputs)
        debug_outputs = ibackend.to_list(debug_outputs)
        constant_inputs = ibackend.to_list(constant_inputs)

        self._n_data_input = len(model_inputs)
        self._n_constant_input = len(constant_inputs)
        self._n_data_output = len(analysis_outputs)
        self._n_debug_output = len(debug_outputs)

        inputs = model_inputs + analysis_inputs + constant_inputs
        outputs = analysis_outputs + debug_outputs

        self._analyzer_model = kmodels.Model(
            inputs=inputs,
            outputs=outputs,
            name=f"{self.__class__.__name__}_analyzer_model",
        )
        self._analyzer_model_done = True

    def _create_analysis(
        self, model: Model, stop_analysis_at_tensors: list[Tensor] = None
    ) -> (
        tuple[list[Tensor]]
        | tuple[list[Tensor], list[Tensor]]
        | tuple[list[Tensor], list[Tensor], list[Tensor]]
    ):
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

    def analyze(
        self,
        X: OptionalList[np.ndarray],
        neuron_selection: OptionalList[int] | None = None,
    ) -> OptionalList[np.ndarray]:
        """
        Same interface as :class:`Analyzer` besides

        :param neuron_selection: If neuron_selection_mode is 'index' this
        should be an integer with the index for the chosen neuron.
        When analyzing batches, this should be a List of integer indices.
        """
        # TODO: what does should mean in docstring?

        if self._analyzer_model_done is False:
            self.create_analyzer_model()

        if neuron_selection is not None and self._neuron_selection_mode != "index":
            raise ValueError(
                f"neuron_selection_mode {self._neuron_selection_mode} doesn't support ",
                "'neuron_selection' parameter.",
            )

        if neuron_selection is None and self._neuron_selection_mode == "index":
            raise ValueError(
                "neuron_selection_mode 'index' expects 'neuron_selection' parameter."
            )

        ret: OptionalList[np.ndarray]
        if self._neuron_selection_mode == "index":
            if neuron_selection is not None:
                batch_size = np.shape(X)[0]
                indices = self._get_neuron_selection_array(neuron_selection, batch_size)
                ret = self._analyzer_model.predict_on_batch([X, indices])
            else:
                raise RuntimeError(
                    'neuron_selection_mode "index" requires neuron_selection.'
                )
        else:
            ret = self._analyzer_model.predict_on_batch(X)

        if self._n_debug_output > 0:
            self._handle_debug_output(ret[-self._n_debug_output :])
            ret = ret[: -self._n_debug_output]

        return ibackend.unpack_singleton(ret)

    def _get_neuron_selection_array(
        self, neuron_selection: OptionalList[int], batch_size: int
    ) -> np.ndarray:
        """Get neuron selection array for neuron_selection_mode "index"."""
        nsa = np.asarray(neuron_selection).flatten()

        # If `nsa` is singleton, repeat it so that it matches the batch size
        if nsa.size == 1:
            nsa = np.repeat(nsa, batch_size)

        # Multiples of batch size are allowed for use with AugmentReduceBase:
        if nsa.size % batch_size != 0:
            raise ValueError(
                f"""`neuron_selection` should be integer or array matching
                batch size {batch_size}. Got: {neuron_selection}"""
            )

        # We prepend a counter for the position in the batch,
        # which will be used by the layer `NeuronSelection`.
        # Using `nsa.size` for compatibility with AugmentReduceBase.
        batch_position_index = np.arange(nsa.size)
        return np.hstack((batch_position_index.reshape((-1, 1)), nsa.reshape((-1, 1))))

    def _get_state(self) -> dict[str, Any]:
        state = super()._get_state()
        state.update({"allow_lambda_layers": self._allow_lambda_layers})
        return state

    @classmethod
    def _state_to_kwargs(cls, state: dict[str, Any]) -> dict[str, Any]:
        allow_lambda_layers = state.pop("allow_lambda_layers")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"allow_lambda_layers": allow_lambda_layers})
        return kwargs
