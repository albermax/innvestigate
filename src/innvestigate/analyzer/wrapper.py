from __future__ import annotations

import numpy as np
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.backend as ibackend
import innvestigate.layers as ilayers
from innvestigate.analyzer.base import AnalyzerBase
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.backend.types import OptionalList, Tensor

__all__ = [
    "WrapperBase",
    "AugmentReduceBase",
    "GaussianSmoother",
    "PathIntegrator",
]


class WrapperBase(AnalyzerBase):
    """Interface for wrappers around analyzers

    This class is the basic interface for wrappers around analyzers.

    :param subanalyzer: The analyzer to be wrapped.
    """

    def __init__(self, subanalyzer: AnalyzerBase, *args, **kwargs):
        if not isinstance(subanalyzer, AnalyzerNetworkBase):
            raise NotImplementedError("Keras-based subanalyzer required.")

        # To simplify serialization, additionaly passed models are popped
        # and the subanalyzer model is passed to `AnalyzerBase`.
        kwargs.pop("model", None)
        super().__init__(subanalyzer._model, *args, **kwargs)

        self._subanalyzer_name = subanalyzer.__class__.__name__
        self._subanalyzer = subanalyzer

    def analyze(self, *args, **kwargs):
        return self._subanalyzer.analyze(*args, **kwargs)

    def _get_state(self) -> dict:
        sa_class_name, sa_state = self._subanalyzer.save()

        state = super()._get_state()
        state.update({"subanalyzer_class_name": sa_class_name})
        state.update({"subanalyzer_state": sa_state})
        return state

    @classmethod
    def _state_to_kwargs(cls, state: dict):
        sa_class_name = state.pop("subanalyzer_class_name")
        sa_state = state.pop("subanalyzer_state")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        subanalyzer = AnalyzerBase.load(sa_class_name, sa_state)
        kwargs.update({"subanalyzer": subanalyzer})
        return kwargs


###############################################################################


class AugmentReduceBase(WrapperBase):
    """Interface for wrappers that augment the input and reduce the analysis.

    This class is an interface for wrappers that:
    * augment the input to the analyzer by creating new samples.
    * reduce the returned analysis to match the initial input shapes.

    :param subanalyzer: The analyzer to be wrapped.
    :param augment_by_n: Number of samples to create.
    """

    def __init__(
        self,
        subanalyzer: AnalyzerNetworkBase,
        *args,
        augment_by_n: int = 2,
        neuron_selection_mode="max_activation",
        **kwargs,
    ):

        if neuron_selection_mode == "max_activation":
            # TODO: find a more transparent way.
            #
            # Since AugmentReduceBase analyzers augment the input,
            # it is possible that the neuron w/ max activation changes.
            # As a workaround, the index of the maximally activated neuron
            # w.r.t. the "unperturbed" input is computed and used in combination
            # with neuron_selection_mode = "index" in the subanalyzer.
            #
            # NOTE:
            # The analyzer will still have neuron_selection_mode = "max_activation"!
            subanalyzer._neuron_selection_mode = "index"

        super().__init__(
            subanalyzer, *args, neuron_selection_mode=neuron_selection_mode, **kwargs
        )

        self._augment_by_n: int = augment_by_n  # number of samples to create

    def create_analyzer_model(self):
        self._subanalyzer.create_analyzer_model()

        if self._subanalyzer._n_debug_output > 0:
            raise NotImplementedError("No debug output at subanalyzer is supported.")

        model = self._subanalyzer._analyzer_model
        if None in model.input_shape[1:]:
            raise ValueError(
                "The input shape for the model needs "
                "to be fully specified (except the batch axis). "
                f"Model input shape is: {model.input_shape}"
            )

        inputs = model.inputs[: self._subanalyzer._n_data_input]
        extra_inputs = model.inputs[self._subanalyzer._n_data_input :]

        # outputs = model.outputs[: self._subanalyzer._n_data_output]
        extra_outputs = model.outputs[self._subanalyzer._n_data_output :]

        if len(extra_outputs) > 0:
            raise Exception("No extra output is allowed with this wrapper.")

        new_inputs = self._augment(ibackend.to_list(inputs))

        augmented_outputs = ibackend.to_list(model(new_inputs + extra_inputs))
        new_outputs = self._reduce(augmented_outputs)
        new_constant_inputs = self._keras_get_constant_inputs()

        inputs = inputs + extra_inputs + new_constant_inputs
        outputs = new_outputs + extra_outputs
        new_model = kmodels.Model(inputs=inputs, outputs=outputs)
        self._subanalyzer._analyzer_model = new_model

    def analyze(
        self, X: OptionalList[np.ndarray], *args, **kwargs
    ) -> OptionalList[np.ndarray]:
        if self._subanalyzer._analyzer_model is None:
            self.create_analyzer_model()

        ns_mode = self._neuron_selection_mode
        if ns_mode == "all":
            return self._subanalyzer.analyze(X, *args, **kwargs)

        # As described in the AugmentReduceBase init,
        # both ns_mode "max_activation" and "index" make use
        # of a subanalyzer using neuron_selection_mode="index".
        if ns_mode == "max_activation":
            # obtain max neuron activations over batch
            pred = self._subanalyzer._model.predict(X)
            indices = np.argmax(pred, axis=1)
        elif ns_mode == "index":
            # TODO: make neuron_selection arg or kwarg, not both
            if args:
                indices = list(args).pop(0)
            else:
                indices = kwargs.pop("neuron_selection")

        if not self._subanalyzer._neuron_selection_mode == "index":
            raise AssertionError(
                'Subanalyzer neuron_selection_mode has to be "index" '
                'when using analyzer with neuron_selection_mode != "all".'
            )
        # broadcast to match augmented samples.
        indices = np.repeat(indices, self._augment_by_n)
        kwargs["neuron_selection"] = indices
        return self._subanalyzer.analyze(X, *args, **kwargs)

    def _keras_get_constant_inputs(self) -> list[Tensor] | None:
        return []

    def _augment(self, Xs: OptionalList[Tensor]) -> list[Tensor]:
        """Augment inputs before analyzing them with subanalyzer."""
        repeat = ilayers.Repeat(self._augment_by_n)
        reshape = ilayers.AugmentationToBatchAxis(self._augment_by_n)
        return [reshape(repeat(X)) for X in ibackend.to_list(Xs)]

    def _reduce(self, Xs: OptionalList[Tensor]) -> list[Tensor]:
        """Reduce input Xs by reshaping and taking the mean along
        the axis of augmentation."""
        reshape = ilayers.AugmentationFromBatchAxis(self._augment_by_n)
        reduce = ilayers.ReduceMean()
        means = [reduce(reshape(X)) for X in ibackend.to_list(Xs)]
        return means

    def _get_state(self):
        state = super()._get_state()
        state.update({"augment_by_n": self._augment_by_n})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        augment_by_n = state.pop("augment_by_n")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"augment_by_n": augment_by_n})
        return kwargs


###############################################################################


class GaussianSmoother(AugmentReduceBase):
    """Wrapper that adds noise to the input and averages over analyses

    This wrapper creates new samples by adding Gaussian noise
    to the input. The final analysis is an average of the returned analyses.

    :param subanalyzer: The analyzer to be wrapped.
    :param noise_scale: The stddev of the applied noise.
    :param augment_by_n: Number of samples to create.
    """

    def __init__(self, subanalyzer, *args, noise_scale: float = 1, **kwargs):
        super().__init__(subanalyzer, *args, **kwargs)
        self._noise_scale = noise_scale

    def _augment(self, Xs: OptionalList[Tensor]) -> list[Tensor]:
        repeat = ilayers.Repeat(self._augment_by_n)
        add_noise = ilayers.AddGaussianNoise()
        reshape = ilayers.AugmentationToBatchAxis(self._augment_by_n)

        ret = [reshape(add_noise(repeat(X))) for X in ibackend.to_list(Xs)]
        return ret

    def _get_state(self):
        state = super()._get_state()
        state.update({"noise_scale": self._noise_scale})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        noise_scale = state.pop("noise_scale")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"noise_scale": noise_scale})
        return kwargs


###############################################################################


class PathIntegrator(AugmentReduceBase):
    """Integrated the analysis along a path

    This analyzer:
    * creates a path from input to reference image.
    * creates `steps` number of intermediate inputs and
      creates an analysis for them.
    * sums the analyses and multiplies them with the input-reference_input.

    This wrapper is used to implement Integrated Gradients.
    We refer to the paper for further information.

    :param subanalyzer: The analyzer to be wrapped.
    :param steps: Number of steps for integration.
    :param reference_inputs: The reference input.
    """

    def __init__(
        self, subanalyzer, *args, steps: int = 16, reference_inputs=0, **kwargs
    ):
        super().__init__(subanalyzer, *args, augment_by_n=steps, **kwargs)

        self._reference_inputs = reference_inputs
        self._keras_constant_inputs: list[Tensor] | None = None

    def _keras_set_constant_inputs(self, inputs: list[Tensor]) -> None:
        tmp = [kbackend.variable(X) for X in inputs]
        self._keras_constant_inputs = [
            klayers.Input(tensor=X, shape=X.shape[1:]) for X in tmp
        ]

    def _keras_get_constant_inputs(self) -> list[Tensor] | None:
        return self._keras_constant_inputs

    def _compute_difference(self, Xs: list[Tensor]) -> list[Tensor]:
        if self._keras_constant_inputs is None:
            inputs = ibackend.broadcast_np_tensors_to_keras_tensors(
                self._reference_inputs, Xs
            )
            self._keras_set_constant_inputs(inputs)

        # Type not Optional anymore as as `_keras_set_constant_inputs` has been called.
        reference_inputs: list[Tensor]
        reference_inputs = self._keras_get_constant_inputs()  # type: ignore
        return [klayers.subtract([x, ri]) for x, ri in zip(Xs, reference_inputs)]

    def _augment(self, Xs):
        difference = self._compute_difference(Xs)
        self._keras_difference = difference
        # Make broadcastable.
        difference = [
            ilayers.Reshape((-1, 1) + kbackend.int_shape(x)[1:])(x) for x in difference
        ]

        # Compute path steps.
        multiply_with_linspace = ilayers.MultiplyWithLinspace(
            0, 1, n=self._augment_by_n, axis=1
        )
        path_steps = [multiply_with_linspace(d) for d in difference]

        reference_inputs = self._keras_get_constant_inputs()
        ret = [klayers.Add()([x, p]) for x, p in zip(reference_inputs, path_steps)]
        ret = [ilayers.Reshape((-1,) + kbackend.int_shape(x)[2:])(x) for x in ret]
        return ret

    def _reduce(self, Xs):
        tmp = super()._reduce(Xs)
        difference = self._keras_difference
        del self._keras_difference

        return [klayers.Multiply()([x, d]) for x, d in zip(tmp, difference)]

    def _get_state(self):
        state = super()._get_state()
        state.update({"reference_inputs": self._reference_inputs})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        reference_inputs = state.pop("reference_inputs")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        # We use steps instead.
        kwargs.update(
            {"reference_inputs": reference_inputs, "steps": kwargs["augment_by_n"]}
        )
        del kwargs["augment_by_n"]
        return kwargs
