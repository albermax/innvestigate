# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import keras.layers
import keras.models


from . import base
from .relevance_based import relevance_rule as lrp_rules
from ..utils.keras import checks as kchecks


__all__ = [
    "DeepTaylor",
    "BoundedDeepTaylor",
]


###############################################################################
###############################################################################
###############################################################################


class DeepTaylor(base.ReverseAnalyzerBase):
    """DeepTaylor for ReLU-networks with unbounded input

    This class implements the DeepTaylor algorithm for neural networks with
    ReLU activation and unbounded input ranges.

    :param model: A Keras model.
    """

    def __init__(self, model, *args, **kwargs):

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "This DeepTaylor implementation only supports ReLU activations.",
            check_type="exception",
        )
        super(DeepTaylor, self).__init__(model, *args, **kwargs)

    def _create_analysis(self, *args, **kwargs):

        def do_nothing(Xs, Ys, As, reverse_state):
            return As

        # Kernel layers.
        self._add_conditional_reverse_mapping(
            lambda l: (kchecks.contains_kernel(l) and
                       kchecks.contains_activation(l)),
            lrp_rules.Alpha1Beta0Rule,
            name="deep_taylor_kernel_w_relu",
        )
        self._add_conditional_reverse_mapping(
            lambda l: (kchecks.contains_kernel(l) and
                       not kchecks.contains_activation(l)),
            lrp_rules.WSquareRule,
            name="deep_taylor_kernel_wo_relu",
        )

        # ReLU Activation layer
        self._add_conditional_reverse_mapping(
            lambda l: (not kchecks.contains_kernel(l) and
                       kchecks.contains_activation(l)),
            self._gradient_reverse_mapping,
            name="deep_taylor_relu",
        )

        # Special layers.
        self._add_conditional_reverse_mapping(
            kchecks.is_batch_normalization_layer,
            do_nothing,
            name="deep_taylor_batch_norm",
        )
        self._add_conditional_reverse_mapping(
            kchecks.is_max_pooling,
            self._gradient_reverse_mapping,
            name="deep_taylor_max_pooling",
        )
        self._add_conditional_reverse_mapping(
            kchecks.is_average_pooling,
            self._gradient_reverse_mapping,
            name="deep_taylor_average_pooling",
        )
        self._add_conditional_reverse_mapping(
            lambda l: isinstance(l, keras.layers.Add),
            # Ignore scaling with 0.5
            self._gradient_reverse_mapping,
            name="deep_taylor_add",
        )
        self._add_conditional_reverse_mapping(
            lambda l: isinstance(l, (
                keras.layers.convolutional.UpSampling1D,
                keras.layers.convolutional.UpSampling2D,
                keras.layers.convolutional.UpSampling3D,
                keras.layers.core.Dropout,
                keras.layers.core.SpatialDropout1D,
                keras.layers.core.SpatialDropout2D,
                keras.layers.core.SpatialDropout3D,
            )),
            self._gradient_reverse_mapping,
            name="deep_taylor_special_layers",
        )

        # Layers w/o transformation
        self._add_conditional_reverse_mapping(
            lambda l: isinstance(l, (
                keras.engine.topology.InputLayer,
                keras.layers.convolutional.Cropping1D,
                keras.layers.convolutional.Cropping2D,
                keras.layers.convolutional.Cropping3D,
                keras.layers.convolutional.ZeroPadding1D,
                keras.layers.convolutional.ZeroPadding2D,
                keras.layers.convolutional.ZeroPadding3D,
                keras.layers.Concatenate,
                keras.layers.core.Flatten,
                keras.layers.core.Masking,
                keras.layers.core.Permute,
                keras.layers.core.RepeatVector,
                keras.layers.core.Reshape,
            )),
            self._gradient_reverse_mapping,
            name="deep_taylor_no_transform",
        )

        return super(DeepTaylor, self)._create_analysis(
            *args, **kwargs)

    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        """
        Block all default mappings.
        """
        raise NotImplementedError(
            "Layer %s not supported." % reverse_state["layer"])

    def _prepare_model(self, model):
        """
        To be theoretically sound Deep-Taylor expects only positive outputs.
        """

        positive_outputs = [keras.layers.ReLU()(x) for x in model.outputs]
        model_with_positive_output = keras.models.Model(
            inputs=model.inputs, outputs=positive_outputs)

        return super(DeepTaylor, self)._prepare_model(
            model_with_positive_output)


class BoundedDeepTaylor(DeepTaylor):
    """DeepTaylor for ReLU-networks with bounded input

    This class implements the DeepTaylor algorithm for neural networks with
    ReLU activation and bounded input ranges.

    :param model: A Keras model.
    :param low: Lowest value of the input range. See Z_B rule.
    :param high: Highest value of the input range. See Z_B rule.
    """

    def __init__(self, model, low=None, high=None, **kwargs):

        if low is None or high is None:
            raise ValueError("The low or high parameter is missing."
                             " Z-B (bounded rule) require both values.")

        self._bounds_low = low
        self._bounds_high = high

        super(BoundedDeepTaylor, self).__init__(
            model, **kwargs)

    def _create_analysis(self, *args, **kwargs):

        low, high = self._bounds_low, self._bounds_high

        class BoundedProxyRule(lrp_rules.BoundedRule):
            def __init__(self, *args, **kwargs):
                super(BoundedProxyRule, self).__init__(
                    *args, low=low, high=high,
                    **kwargs)

        self._add_conditional_reverse_mapping(
            lambda l: kchecks.is_input_layer(l) and kchecks.contains_kernel(l),
            BoundedProxyRule,
            name="deep_taylor_first_layer_bounded",
            priority=10,  # do first
        )

        return super(BoundedDeepTaylor, self)._create_analysis(
            *args, **kwargs)
