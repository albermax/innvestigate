from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import keras.backend as K
import keras.layers
import keras.models
import numpy as np

import innvestigate.layers as ilayers
import innvestigate.utils as iutils
import innvestigate.utils.keras.checks as kchecks
from innvestigate.analyzer.base import AnalyzerBase
from innvestigate.utils.types import Layer, LayerCheck, Model, OptionalList, Tensor

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
        self._special_helper_layers: List[Layer] = []  # added for _reverse_mapping
        self._analysis_inputs: Optional[List[Tensor]] = None
        self._n_data_input: int = 0
        self._n_constant_input: int = 0
        self._n_data_output: int = 0
        self._n_debug_output: int = 0

    def _add_model_softmax_check(self) -> None:
        """
        Adds check that prevents models from containing a softmax.
        """
        contains_softmax: LayerCheck = lambda layer: kchecks.contains_activation(
            layer, activation="softmax"
        )
        self._add_model_check(
            check=contains_softmax,
            message="This analysis method does not support softmax layers.",
            check_type="exception",
        )

    def _add_lambda_layers_check(self) -> None:
        check_lambda_layers: LayerCheck = lambda layer: (
            not self._allow_lambda_layers
            and isinstance(layer, keras.layers.core.Lambda)
        )
        self._add_model_check(
            check=check_lambda_layers,
            message=(
                "Lamda layers are not allowed. "
                "To force use set 'allow_lambda_layers' parameter."
            ),
            check_type="exception",
        )

    def _prepare_model(self, model: Model) -> Tuple[Model, List[Tensor], List[Tensor]]:
        """
        Prepares the model to analyze before it gets actually analyzed.

        This class adds the code to select a specific output neuron.
        """
        neuron_selection_mode: str
        model_inputs: OptionalList[Tensor]
        model_output: OptionalList[Tensor]
        analysis_inputs: List[Tensor]
        stop_analysis_at_tensors: List[Tensor]

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
            inn_max = ilayers.Max(name="iNNvestigate_max")
            model_output = inn_max(model_output)
            self._special_helper_layers.append(inn_max)

        elif neuron_selection_mode == "index":
            # Creates a placeholder tensor when `dtype` is passed.
            neuron_indexing: Layer = keras.layers.Input(
                batch_shape=[None, None],
                dtype=np.int32,
                name="iNNvestigate_neuron_indexing",
            )
            # TODO: what does _keras_history[0] do?
            self._special_helper_layers.append(neuron_indexing._keras_history[0])
            analysis_inputs.append(neuron_indexing)
            # The indexing tensor should not be analyzed.
            stop_analysis_at_tensors.append(neuron_indexing)

            inn_gather = ilayers.GatherND(name="iNNvestigate_gather_nd")
            model_output = inn_gather(model_output + [neuron_indexing])
            self._special_helper_layers.append(inn_gather)
        elif neuron_selection_mode == "all":
            pass
        else:
            raise NotImplementedError()

        model = keras.models.Model(
            inputs=model_inputs + analysis_inputs, outputs=model_output
        )
        return model, analysis_inputs, stop_analysis_at_tensors

    def create_analyzer_model(self) -> None:
        """
        Creates the analyze functionality. If not called beforehand
        it will be called by :func:`analyze`.
        """
        model_inputs = self._model.inputs
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

        self._analyzer_model_done = True

    def _create_analysis(
        self, model: Model, stop_analysis_at_tensors: List[Tensor] = None
    ) -> Union[
        Tuple[List[Tensor]],
        Tuple[List[Tensor], List[Tensor]],
        Tuple[List[Tensor], List[Tensor], List[Tensor]],
    ]:
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
        neuron_selection: Optional[int] = None,
    ) -> OptionalList[np.ndarray]:
        """
        Same interface as :class:`Analyzer` besides

        :param neuron_selection: If neuron_selection_mode is 'index' this
        should be an integer with the index for the chosen neuron.
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

        X = iutils.to_list(X)

        ret: OptionalList[np.ndarray]
        if self._neuron_selection_mode == "index":
            if neuron_selection is not None:
                # TODO: document how this works
                selection = self._get_neuron_selection_array(X, neuron_selection)
                ret = self._analyzer_model.predict_on_batch(X + [selection])
            else:
                raise RuntimeError(
                    'neuron_selection_mode "index" requires neuron_selection.'
                )
        else:
            ret = self._analyzer_model.predict_on_batch(X)

        if self._n_debug_output > 0:
            self._handle_debug_output(ret[-self._n_debug_output :])
            ret = ret[: -self._n_debug_output]

        return iutils.unpack_singleton(ret)

    def _get_neuron_selection_array(
        self, X: List[np.ndarray], neuron_selection: int
    ) -> np.ndarray:
        """Get neuron selection array for neuron_selection_mode "index"."""
        # TODO: detailed documentation on how this selects neurons

        nsa = np.asarray(neuron_selection).flatten()  # singleton ndarray

        # is 'nsa' is singleton, repeat it so that it matches number of rows of X
        if nsa.size == 1:
            nsa = np.repeat(nsa, len(X[0]))

        # Add first axis indices for gather_nd
        nsa = np.hstack((np.arange(len(nsa)).reshape((-1, 1)), nsa.reshape((-1, 1))))
        return nsa

    def _get_state(self) -> Dict[str, Any]:
        state = super()._get_state()
        state.update({"allow_lambda_layers": self._allow_lambda_layers})
        return state

    @classmethod
    def _state_to_kwargs(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        allow_lambda_layers = state.pop("allow_lambda_layers")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"allow_lambda_layers": allow_lambda_layers})
        return kwargs
