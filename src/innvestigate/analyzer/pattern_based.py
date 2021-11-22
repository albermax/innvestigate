from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import tensorflow.keras.activations as kactivations
import tensorflow.keras.layers as klayers

import innvestigate.tools as itools
import innvestigate.utils.keras as ikeras
import innvestigate.utils.keras.backend as ibackend
import innvestigate.utils.keras.checks as ichecks
import innvestigate.utils.keras.graph as igraph
from innvestigate.analyzer.base import OneEpochTrainerMixin
from innvestigate.analyzer.reverse_base import ReverseAnalyzerBase

__all__ = [
    "PatternNet",
    "PatternAttribution",
]


SUPPORTED_LAYER_PATTERNNET = (
    klayers.InputLayer,
    klayers.Conv2D,
    klayers.Dense,
    klayers.Dropout,
    klayers.Flatten,
    klayers.Masking,
    klayers.Permute,
    klayers.Reshape,
    klayers.Concatenate,
    klayers.GlobalMaxPooling1D,
    klayers.GlobalMaxPooling2D,
    klayers.GlobalMaxPooling3D,
    klayers.MaxPooling1D,
    klayers.MaxPooling2D,
    klayers.MaxPooling3D,
)


class PatternNetReverseKernelLayer(igraph.ReverseMappingBase):
    """
    PatternNet backward mapping for layers with kernels.

    Applies the (filter) weights on the forward pass and
    on the backward pass applies the gradient computation
    where the filter weights are replaced with the patterns.
    """

    def __init__(self, layer, _state, pattern):
        config = layer.get_config()

        # Layer can contain a kernel and an activation.
        # Split layers in a kernel layer and an activation layer.
        activation = None
        if "activation" in config:
            activation = config["activation"]
            config["activation"] = None
        self._act_layer = klayers.Activation(
            activation, name="reversed_act_%s" % config["name"]
        )
        self._filter_layer = igraph.copy_layer_wo_activation(
            layer, name_template="reversed_filter_%s"
        )

        # Replace filter/kernel weights with patterns.
        filter_weights = layer.get_weights()
        # Assume that only one weight has a corresponding pattern.
        # E.g., biases have no pattern.
        tmp = [pattern.shape == x.shape for x in filter_weights]
        if np.sum(tmp) != 1:
            raise Exception("Cannot match pattern to filter.")
        filter_weights[np.argmax(tmp)] = pattern
        self._pattern_layer = igraph.copy_layer_wo_activation(
            layer, name_template="reversed_pattern_%s", weights=filter_weights
        )

    def apply(self, Xs, _Ys, reversed_Ys, _reverse_state: Dict):
        # Reapply the prepared layers.
        act_Xs = ikeras.apply(self._filter_layer, Xs)
        act_Ys = ikeras.apply(self._act_layer, act_Xs)
        pattern_Ys = ikeras.apply(self._pattern_layer, Xs)

        # Layers that apply the backward pass.
        grad_act = ilayers.GradientWRT(len(act_Xs))
        grad_pattern = ilayers.GradientWRT(len(Xs))

        # First step: propagate through the activation layer.
        # Workaround for linear activations.
        linear_activations = [None, kactivations.get("linear")]
        if self._act_layer.activation in linear_activations:
            tmp = reversed_Ys
        else:
            # if linear activation this behaves strange
            tmp = iutils.to_list(grad_act(act_Xs + act_Ys + reversed_Ys))

        # Second step: propagate through the pattern layer.
        return grad_pattern(Xs + pattern_Ys + tmp)


class PatternNet(OneEpochTrainerMixin, ReverseAnalyzerBase):
    """PatternNet analyzer.

    Applies the "PatternNet" algorithm to analyze the model's predictions.

    :param model: A Keras model.
    :param patterns: Pattern computed by
      :class:`innvestigate.tools.PatternComputer`. If None :func:`fit` needs
      to be called.
    :param allow_lambda_layers: Approximate lambda layers with the gradient.
    :param reverse_project_bottleneck_layers: Project the analysis vector into
      range [-1, +1]. (default: True)
    """

    def __init__(self, model, patterns=None, pattern_type=None, **kwargs):
        super().__init__(model, **kwargs)

        # Add and run model checks
        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not ichecks.only_relu_activation(layer),
            ("PatternNet is not well defined for networks with non-ReLU activations."),
            check_type="warning",
        )
        self._add_model_check(
            lambda layer: not ichecks.is_convnet_layer(layer),
            ("PatternNet is only well defined for convolutional neural networks."),
            check_type="warning",
        )
        self._add_model_check(
            lambda layer: not isinstance(layer, SUPPORTED_LAYER_PATTERNNET),
            ("PatternNet is only well defined for conv2d/max-pooling/dense layers."),
            check_type="exception",
        )
        self._do_model_checks()

        self._patterns = patterns
        if self._patterns is not None:
            # copy pattern references
            self._patterns = list(patterns)
        self._pattern_type = pattern_type

        # Pattern projections can lead to +-inf value with long networks.
        # We are only interested in the direction, therefore it is save to
        # Prevent this by projecting the values in bottleneck layers to +-1.
        if not kwargs.get("reverse_project_bottleneck_layers", True):
            warnings.warn(
                "The standard setting for 'reverse_project_bottleneck_layers'"
                "is overwritten."
            )
        else:
            kwargs["reverse_project_bottleneck_layers"] = True

    def _get_pattern_for_layer(self, layer, _state):
        layers = [
            l
            for l in igraph.get_model_layers(self._model)
            if ichecks.contains_kernel(l)
        ]

        return self._patterns[layers.index(layer)]

    def _prepare_pattern(self, _layer, _state, pattern):
        """ ""Prepares a pattern before it is set in the back-ward pass."""
        return pattern

    def _create_analysis(self, *args, **kwargs):

        # Apply the pattern mapping on all layers that contain a kernel.
        def create_kernel_layer_mapping(layer, state):
            pattern = self._get_pattern_for_layer(layer, state)
            pattern = self._prepare_pattern(layer, state, pattern)
            mapping_obj = PatternNetReverseKernelLayer(layer, state, pattern)
            return mapping_obj.apply

        self._add_conditional_reverse_mapping(
            ichecks.contains_kernel,
            create_kernel_layer_mapping,
            name="patternnet_kernel_layer_mapping",
        )

        return super()._create_analysis(*args, **kwargs)

    def _fit_generator(
        self,
        generator,
        steps_per_epoch=None,
        epochs=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
        disable_no_training_warning=None,
        **kwargs
    ):
        # TODO: implement epochs

        pattern_type = self._pattern_type
        if pattern_type is None:
            pattern_type = "relu"

        if isinstance(pattern_type, (list, tuple)):
            raise ValueError("Only one pattern type allowed. Please pass a string.")

        computer = itools.PatternComputer(
            self._model, pattern_type=pattern_type, **kwargs
        )

        self._patterns = computer.compute_generator(
            generator,
            steps_per_epoch=steps_per_epoch,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=verbose,
        )

    def _get_state(self):
        state = super()._get_state()
        state.update({"patterns": self._patterns, "pattern_type": self._pattern_type})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        patterns = state.pop("patterns")
        pattern_type = state.pop("pattern_type")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"patterns": patterns, "pattern_type": pattern_type})
        return kwargs


class PatternAttribution(PatternNet):
    """PatternAttribution analyzer.

    Applies the "PatternNet" algorithm to analyze the model's predictions.

    :param model: A Keras model.
    :param patterns: Pattern computed by
      :class:`innvestigate.tools.PatternComputer`. If None :func:`fit` needs
      to be called.
    :param allow_lambda_layers: Approximate lambda layers with the gradient.
    :param reverse_project_bottleneck_layers: Project the analysis vector into
      range [-1, +1]. (default: True)
    """

    def _prepare_pattern(self, layer, state, pattern):
        weights = layer.get_weights()
        tmp = [pattern.shape == x.shape for x in weights]
        if np.sum(tmp) != 1:
            raise Exception("Cannot match pattern to kernel.")
        weight = weights[np.argmax(tmp)]
        return np.multiply(pattern, weight)
