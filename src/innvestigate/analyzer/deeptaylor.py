from __future__ import annotations

from typing import Any

import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.analyzer.relevance_based.relevance_rule as lrp_rules
import innvestigate.backend.checks as ichecks
import innvestigate.backend.graph as igraph
from innvestigate.analyzer.reverse_base import ReverseAnalyzerBase
from innvestigate.backend.types import Model

__all__ = [
    "DeepTaylor",
    "BoundedDeepTaylor",
]


class DeepTaylor(ReverseAnalyzerBase):
    """DeepTaylor for ReLU-networks with unbounded input

    This class implements the DeepTaylor algorithm for neural networks with
    ReLU activation and unbounded input ranges.

    :param model: A Keras model.
    """

    def __init__(self, model: Model, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        # Add and run model checks
        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not ichecks.only_relu_activation(layer),
            "This DeepTaylor implementation only supports ReLU activations.",
            check_type="exception",
        )
        self._do_model_checks()

    def _create_analysis(self, *args: Any, **kwargs: Any):

        # Kernel layers.
        self._add_conditional_reverse_mapping(
            lambda l: (ichecks.contains_kernel(l) and ichecks.contains_activation(l)),
            lrp_rules.Alpha1Beta0IgnoreBiasRule,
            name="deep_taylor_kernel_w_relu",
        )
        self._add_conditional_reverse_mapping(
            lambda l: (
                ichecks.contains_kernel(l) and not ichecks.contains_activation(l)
            ),
            lrp_rules.WSquareRule,
            name="deep_taylor_kernel_wo_relu",
        )

        # ReLU Activation layer
        self._add_conditional_reverse_mapping(
            lambda l: (
                not ichecks.contains_kernel(l) and ichecks.only_relu_activation(l)
            ),
            self._gradient_reverse_mapping,
            name="deep_taylor_relu",
        )

        # Assume conv layer beforehand -> unbounded
        bn_mapping = igraph.apply_mapping_to_fused_bn_layer(
            lrp_rules.WSquareRule,
            fuse_mode="one_linear",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_batch_normalization_layer,
            bn_mapping,
            name="deep_taylor_batch_norm",
        )
        # Special layers.
        self._add_conditional_reverse_mapping(
            ichecks.is_max_pooling,
            self._gradient_reverse_mapping,
            name="deep_taylor_max_pooling",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_average_pooling,
            self._gradient_reverse_mapping,
            name="deep_taylor_average_pooling",
        )
        self._add_conditional_reverse_mapping(
            lambda l: isinstance(l, klayers.Add),
            # Ignore scaling with 0.5
            self._gradient_reverse_mapping,
            name="deep_taylor_add",
        )
        self._add_conditional_reverse_mapping(
            lambda l: isinstance(
                l,
                (
                    klayers.UpSampling1D,
                    klayers.UpSampling2D,
                    klayers.UpSampling3D,
                    klayers.Dropout,
                    klayers.SpatialDropout1D,
                    klayers.SpatialDropout2D,
                    klayers.SpatialDropout3D,
                ),
            ),
            self._gradient_reverse_mapping,
            name="deep_taylor_special_layers",
        )

        # Layers w/o transformation
        self._add_conditional_reverse_mapping(
            lambda l: isinstance(
                l,
                (
                    klayers.InputLayer,
                    klayers.Cropping1D,
                    klayers.Cropping2D,
                    klayers.Cropping3D,
                    klayers.ZeroPadding1D,
                    klayers.ZeroPadding2D,
                    klayers.ZeroPadding3D,
                    klayers.Concatenate,
                    klayers.Flatten,
                    klayers.Masking,
                    klayers.Permute,
                    klayers.RepeatVector,
                    klayers.Reshape,
                ),
            ),
            self._gradient_reverse_mapping,
            name="deep_taylor_no_transform",
        )

        return super()._create_analysis(*args, **kwargs)

    def _default_reverse_mapping(self, _Xs, _Ys, _reversed_Ys, reverse_state):
        """
        Block all default mappings.
        """
        raise NotImplementedError(f"""Layer {reverse_state["layer"]} not supported.""")

    def _prepare_model(self, model):
        """
        To be theoretically sound Deep-Taylor expects only positive outputs.
        """

        positive_outputs = [klayers.ReLU()(x) for x in model.outputs]
        model_with_positive_output = kmodels.Model(
            inputs=model.inputs, outputs=positive_outputs
        )

        return super()._prepare_model(model_with_positive_output)


class BoundedDeepTaylor(DeepTaylor):
    """DeepTaylor for ReLU-networks with bounded input

    This class implements the DeepTaylor algorithm for neural networks with
    ReLU activation and bounded input ranges.

    :param model: A Keras model.
    :param low: Lowest value of the input range. See Z_B rule.
    :param high: Highest value of the input range. See Z_B rule.
    """

    def __init__(self, model, low=None, high=None, **kwargs):
        super().__init__(model, **kwargs)

        if low is None or high is None:
            raise ValueError(
                "The low or high parameter is missing. "
                "Z-B (bounded rule) require both values."
            )

        self._bounds_low = low
        self._bounds_high = high

    def _create_analysis(self, *args, **kwargs):

        low, high = self._bounds_low, self._bounds_high

        class BoundedProxyRule(lrp_rules.BoundedRule):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, low=low, high=high, **kwargs)

        self._add_conditional_reverse_mapping(
            lambda l: ichecks.is_input_layer(l) and ichecks.contains_kernel(l),
            BoundedProxyRule,
            name="deep_taylor_first_layer_bounded",
            priority=10,  # do first
        )

        return super()._create_analysis(*args, **kwargs)

    def _get_state(self):
        state = super()._get_state()
        state.update({"low": self._bounds_low, "high": self._bounds_high})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        low = state.pop("low")
        high = state.pop("high")

        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"low": low, "high": high})
        return kwargs
