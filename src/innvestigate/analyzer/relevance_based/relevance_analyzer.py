from __future__ import annotations

import inspect

import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers

import innvestigate.analyzer.relevance_based.relevance_rule as rrule
import innvestigate.analyzer.relevance_based.utils as rutils
import innvestigate.backend as ibackend
import innvestigate.backend.checks as ichecks
import innvestigate.backend.graph as igraph
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.analyzer.reverse_base import ReverseAnalyzerBase
from innvestigate.backend.types import Layer, LayerCheck, Model, OptionalList, Tensor

__all__ = [
    "BaselineLRPZ",
    "LRP",
    "LRP_RULES",
    "LRPZ",
    "LRPEpsilon",
    "LRPWSquare",
    "LRPFlat",
    "LRPAlphaBeta",
    "LRPAlpha2Beta1",
    "LRPAlpha2Beta1IgnoreBias",
    "LRPAlpha1Beta0",
    "LRPAlpha1Beta0IgnoreBias",
    "LRPZPlus",
    "LRPZPlusFast",
    "LRPSequentialPresetA",
    "LRPSequentialPresetB",
    "LRPSequentialPresetAFlat",
    "LRPSequentialPresetBFlat",
    "LRPSequentialPresetBFlatUntilIdx",
]


###############################################################################
BASELINE_LRPZ_LAYERS = (
    klayers.InputLayer,
    klayers.Conv1D,
    klayers.Conv2D,
    klayers.Conv2DTranspose,
    klayers.Conv3D,
    klayers.Conv3DTranspose,
    klayers.Cropping1D,
    klayers.Cropping2D,
    klayers.Cropping3D,
    klayers.SeparableConv1D,
    klayers.SeparableConv2D,
    klayers.UpSampling1D,
    klayers.UpSampling2D,
    klayers.UpSampling3D,
    klayers.ZeroPadding1D,
    klayers.ZeroPadding2D,
    klayers.ZeroPadding3D,
    klayers.Activation,
    klayers.ActivityRegularization,
    klayers.Dense,
    klayers.Dropout,
    klayers.Flatten,
    klayers.Lambda,
    klayers.Masking,
    klayers.Permute,
    klayers.RepeatVector,
    klayers.Reshape,
    klayers.SpatialDropout1D,
    klayers.SpatialDropout2D,
    klayers.SpatialDropout3D,
    klayers.LocallyConnected1D,
    klayers.LocallyConnected2D,
    klayers.Add,
    klayers.Concatenate,
    klayers.Dot,
    klayers.Maximum,
    klayers.Minimum,
    klayers.Subtract,
    klayers.AlphaDropout,
    klayers.GaussianDropout,
    klayers.GaussianNoise,
    klayers.BatchNormalization,
    klayers.GlobalMaxPooling1D,
    klayers.GlobalMaxPooling2D,
    klayers.GlobalMaxPooling3D,
    klayers.MaxPooling1D,
    klayers.MaxPooling2D,
    klayers.MaxPooling3D,
)


class BaselineLRPZ(AnalyzerNetworkBase):
    """LRPZ analyzer - for testing purpose only.

    Applies the "LRP-Z" algorithm to analyze the model.
    Based on the gradient times the input formula.
    **This formula holds only for ReLU/MaxPooling networks, for which
    LRP-Z collapses into the stated formula.**

    :param model: A Keras model.
    """

    def __init__(self, model: Model, **kwargs):
        # Inside function to not break import if Keras changes.

        super().__init__(model, **kwargs)

        # Add and run model checks
        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not ichecks.only_relu_activation(layer),
            "BaselineLRPZ only works with  ReLU activations.",
            check_type="exception",
        )
        self._add_model_check(
            lambda layer: not isinstance(layer, BASELINE_LRPZ_LAYERS),
            "BaselineLRPZ only works with a predefined set of layers.",
            check_type="exception",
        )
        self._do_model_checks()

    def _create_analysis(
        self, model: Model, stop_analysis_at_tensors: list[Tensor] = None
    ):
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        tensors_to_analyze = [
            x
            for x in ibackend.to_list(model.inputs)
            if x not in stop_analysis_at_tensors
        ]
        gradients = ibackend.to_list(
            kbackend.gradients(model.outputs[0], tensors_to_analyze)
        )
        return [
            klayers.Multiply()([i, g]) for i, g in zip(tensors_to_analyze, gradients)
        ]


###############################################################################

# Utility list enabling name mappings via string
LRP_RULES: dict = {
    "Z": rrule.ZRule,
    "Epsilon": rrule.EpsilonRule,
    "WSquare": rrule.WSquareRule,
    "Flat": rrule.FlatRule,
    "AlphaBeta": rrule.AlphaBetaRule,
    "AlphaBetaIgnoreBias": rrule.AlphaBetaIgnoreBiasRule,
    "Alpha2Beta1": rrule.Alpha2Beta1Rule,
    "Alpha2Beta1IgnoreBias": rrule.Alpha2Beta1IgnoreBiasRule,
    "Alpha1Beta0": rrule.Alpha1Beta0Rule,
    "Alpha1Beta0IgnoreBias": rrule.Alpha1Beta0IgnoreBiasRule,
    "ZPlus": rrule.ZPlusRule,
    "ZPlusFast": rrule.ZPlusFastRule,
    "Bounded": rrule.BoundedRule,
}


class EmbeddingReverseLayer(igraph.ReverseMappingBase):
    def __init__(self, _layer, _state):
        # TODO: implement rule support.
        pass

    def apply(self, _Xs, _Ys, Rs, _reverse_state: dict):
        # the embedding layer outputs for an (indexed) input a vector.
        # thus, in the relevance backward pass, the embedding layer receives
        # relevances Rs corresponding to those vectors.
        # Due to the 1:1 relationship between input index and output mapping vector,
        # the relevance backward pass can be realized by pooling relevances
        # over the vector axis.

        # relevances are given shaped [batch_size, sequence_length, embedding_dims]
        pool_relevance = klayers.Lambda(lambda x: kbackend.sum(x, axis=-1))
        return [pool_relevance(R) for R in Rs]


class BatchNormalizationReverseLayer(igraph.ReverseMappingBase):
    """Special BN handler that applies the Z-Rule"""

    def __init__(self, layer, _state):
        config = layer.get_config()

        self._center = config["center"]
        self._scale = config["scale"]
        self._axis = config["axis"]

        self._mean = layer.moving_mean
        self._std = layer.moving_variance
        if self._center:
            self._beta = layer.beta

        # TODO: implement rule support.
        #   for BatchNormalization -> [BNEpsilon, BNAlphaBeta, BNIgnore]
        # super().__init__(layer, state)
        # how to do this:
        # super.__init__ calls select_rule and sets a self._rule class
        # check if isinstance(self_rule, EpsiloneRule), then reroute
        # to BatchNormEpsilonRule. Not pretty, but should work.

    def apply(self, Xs, Ys, Rs, _reverse_state: dict):
        input_shape = [kbackend.int_shape(x) for x in Xs]
        if len(input_shape) != 1:
            # extend below lambda layers towards multiple parameters.
            raise ValueError(
                "BatchNormalizationReverseLayer expects Xs with length 1, "
                f"but was len(Xs)={len(Xs)}"
            )
        input_shape = input_shape[0]

        # prepare broadcasting shape for layer parameters
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self._axis] = input_shape[self._axis]
        broadcast_shape[0] = -1

        # reweight relevances as
        #        x * (y - beta)     R
        # Rin = ---------------- * ----
        #           x - mu          y
        # batch norm can be considered as 3 distinct layers of subtraction,
        # multiplication and then addition. The multiplicative scaling layer
        # has no effect on LRP and functions as a linear activation layer

        minus_mu = klayers.Lambda(
            lambda x: x - kbackend.reshape(self._mean, broadcast_shape)
        )
        minus_beta = klayers.Lambda(
            lambda x: x - kbackend.reshape(self._beta, broadcast_shape)
        )
        prepare_div = klayers.Lambda(
            lambda x: x
            + (kbackend.cast(kbackend.greater_equal(x, 0), kbackend.floatx()) * 2 - 1)
            * kbackend.epsilon()
        )

        x_minus_mu = ibackend.apply(minus_mu, Xs)
        if self._center:
            y_minus_beta = ibackend.apply(minus_beta, Ys)
        else:
            y_minus_beta = Ys

        numerator = [
            klayers.Multiply()([x, ymb, r]) for x, ymb, r in zip(Xs, y_minus_beta, Rs)
        ]
        denominator = [klayers.Multiply()([xmm, y]) for xmm, y in zip(x_minus_mu, Ys)]

        return [
            ibackend.safe_divide(n, prepare_div(d))
            for n, d in zip(numerator, denominator)
        ]


class AddReverseLayer(igraph.ReverseMappingBase):
    """Special Add layer handler that applies the Z-Rule"""

    def __init__(self, layer, _state):
        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, name_template="reversed_kernel_%s"
        )

        # TODO: implement rule support.
        # super().__init__(layer, state)

    def apply(self, Xs, _Ys, Rs, _reverse_state: dict):
        # The outputs of the pooling operation at each location
        # is the sum of its inputs.
        # The forward message must be known in this case,
        # and are the inputs for each pooling thing.
        # The gradient is 1 for each output-to-input connection,
        # which corresponds to the "weights" of the layer.
        # It should thus be sufficient to reweight the relevances
        # and do a gradient_wrt
        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]

        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]


class AveragePoolingReverseLayer(igraph.ReverseMappingBase):
    """Special AveragePooling handler that applies the Z-Rule"""

    def __init__(self, layer, _state):
        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, name_template="reversed_kernel_%s"
        )

        # TODO: implement rule support.
        # super().__init__(layer, state)

    def apply(self, Xs, _Ys, Rs, reverse_state: dict):
        # The outputs of the pooling operation at each location
        # is the sum of its inputs.
        # The forward message must be known in this case,
        # and are the inputs for each pooling thing.
        # The gradient is 1 for each output-to-input connection,
        # which corresponds to the "weights" of the layer.
        # It should thus be sufficient to reweight the relevances
        # and do a gradient_wrt
        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]

        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]


class LRP(ReverseAnalyzerBase):
    """
    Base class for LRP-based model analyzers


    :param model: A Keras model.

    :param rule: A rule can be a  string or a Rule object, lists thereof or
      a list of conditions [(Condition, Rule), ... ]
      gradient.

    :param input_layer_rule: either a Rule object, atuple of (low, high)
      the min/max pixel values of the inputs
    :param bn_layer_rule: either a Rule object or None.
      None means dedicated BN rule will be applied.
    """

    def __init__(
        self,
        model,
        *args,
        rule=None,
        input_layer_rule=None,
        until_layer_idx=None,
        until_layer_rule=None,
        bn_layer_rule=None,
        bn_layer_fuse_mode: str = "one_linear",
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)

        self._input_layer_rule = input_layer_rule
        self._until_layer_rule = until_layer_rule
        self._until_layer_idx = until_layer_idx
        self._bn_layer_rule = bn_layer_rule
        self._bn_layer_fuse_mode = bn_layer_fuse_mode

        if self._bn_layer_fuse_mode not in ["one_linear", "two_linear"]:
            raise ValueError(f"Unknown _bn_layer_fuse_mode {self._bn_layer_fuse_mode}")

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not ichecks.is_convnet_layer(layer),
            "LRP is only tested for convolutional neural networks.",
            check_type="warning",
        )

        # TODO: refactor rule type checking into separate function
        # check if rule was given explicitly.
        # rule can be a string, a list (of strings) or
        # a list of conditions [(Condition, Rule), ... ] for each layer.
        if rule is None:
            raise ValueError("Need LRP rule(s).")

        if isinstance(rule, list):
            self._rule = list(rule)
        else:
            self._rule = rule

        if isinstance(rule, str) or (
            inspect.isclass(rule) and issubclass(rule, igraph.ReverseMappingBase)
        ):  # NOTE: All LRP rules inherit from igraph.ReverseMappingBase
            # the given rule is a single string or single rule implementing cla ss
            use_conditions = True
            rules = [(lambda _: True, rule)]

        elif not isinstance(rule[0], tuple):
            # rule list of rule strings or classes
            use_conditions = False
            rules = list(rule)
        else:
            # rule is list of conditioned rules
            use_conditions = True
            rules = rule

        # apply rule to first self._until_layer_idx layers
        if self._until_layer_rule is not None and self._until_layer_idx is not None:
            for i in range(self._until_layer_idx + 1):
                is_at_idx: LayerCheck = lambda layer: ichecks.is_layer_at_idx(layer, i)
                rules.insert(0, (is_at_idx, self._until_layer_rule))

        # create a BoundedRule for input layer handling from given tuple
        if self._input_layer_rule is not None:
            input_layer_rule = self._input_layer_rule
            if isinstance(input_layer_rule, tuple):
                low, high = input_layer_rule

                class BoundedProxyRule(rrule.BoundedRule):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, low=low, high=high, **kwargs)

                input_layer_rule = BoundedProxyRule

            if use_conditions is True:
                rules.insert(0, (ichecks.is_input_layer, input_layer_rule))
            else:
                rules.insert(0, input_layer_rule)

        self._rules_use_conditions = use_conditions
        self._rules = rules

    def create_rule_mapping(self, layer: Layer, reverse_state: dict):
        if self._rules_use_conditions is True:
            for condition, rule in self._rules:
                if condition(layer):
                    rule_class = rule
                    break
        else:
            rule_class = self._rules.pop()

        if rule_class is None:
            raise Exception(f"No rule applies to layer {layer}")

        if isinstance(rule_class, str):
            rule_class = LRP_RULES[rule_class]
        rule = rule_class(layer, reverse_state)

        return rule.apply

    def _create_analysis(self, *args, **kwargs):
        ###################################################################
        # Functionality responible for backwards rule selection below  ####
        ###################################################################

        # default backward hook
        self._add_conditional_reverse_mapping(
            ichecks.contains_kernel,
            self.create_rule_mapping,
            name="lrp_layer_with_kernel_mapping",
        )

        # specialized backward hooks.
        # TODO: add ReverseLayer class handling layers without kernel: Add and AvgPool
        bn_layer_rule = self._bn_layer_rule

        if bn_layer_rule is None:
            # TODO (alber): get rid of this option!
            # alternatively a default rule should be applied.
            bn_mapping = BatchNormalizationReverseLayer
        else:
            if isinstance(bn_layer_rule, str):
                bn_layer_rule = LRP_RULES[bn_layer_rule]

            bn_mapping = igraph.apply_mapping_to_fused_bn_layer(
                bn_layer_rule,
                fuse_mode=self._bn_layer_fuse_mode,
            )
        self._add_conditional_reverse_mapping(
            ichecks.is_batch_normalization_layer,
            bn_mapping,
            name="lrp_batch_norm_mapping",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_average_pooling,
            AveragePoolingReverseLayer,
            name="lrp_average_pooling_mapping",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_add_layer,
            AddReverseLayer,
            name="lrp_add_layer_mapping",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_embedding_layer,
            EmbeddingReverseLayer,
            name="lrp_embedding_mapping",
        )

        # FINALIZED constructor.
        return super()._create_analysis(*args, **kwargs)

    def _default_reverse_mapping(
        self,
        Xs: OptionalList[Tensor],
        Ys: OptionalList[Tensor],
        reversed_Ys: OptionalList[Tensor],
        reverse_state: dict,
    ):
        # default_return_layers = [klayers.Activation]# TODO extend
        if (
            len(Xs) == len(Ys)
            and isinstance(reverse_state["layer"], (klayers.Activation,))
            and all(
                kbackend.int_shape(x) == kbackend.int_shape(y) for x, y in zip(Xs, Ys)
            )
        ):
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            return reversed_Ys

        # This branch covers:
        # MaxPooling, Max, Flatten, Reshape, Concatenate, Cropping
        return self._gradient_reverse_mapping(Xs, Ys, reversed_Ys, reverse_state)

    ######################################
    # End of Rule Selection Business. ####
    ######################################

    def _get_state(self):
        state = super()._get_state()
        state.update({"rule": self._rule})
        state.update({"input_layer_rule": self._input_layer_rule})
        state.update({"bn_layer_rule": self._bn_layer_rule})
        state.update({"bn_layer_fuse_mode": self._bn_layer_fuse_mode})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        rule = state.pop("rule")
        input_layer_rule = state.pop("input_layer_rule")
        bn_layer_rule = state.pop("bn_layer_rule")
        bn_layer_fuse_mode = state.pop("bn_layer_fuse_mode")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update(
            {
                "rule": rule,
                "input_layer_rule": input_layer_rule,
                "bn_layer_rule": bn_layer_rule,
                "bn_layer_fuse_mode": bn_layer_fuse_mode,
            }
        )
        return kwargs


###############################################################################
# ANALYZER CLASSES AND PRESETS ################################################
###############################################################################


class _LRPFixedParams(LRP):
    @classmethod
    def _state_to_kwargs(cls, state):
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        del kwargs["rule"]
        del kwargs["bn_layer_rule"]
        return kwargs


class LRPZ(_LRPFixedParams):
    """LRP-analyzer that uses the LRP-Z rule"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, rule="Z", bn_layer_rule="Z", **kwargs)
        self._do_model_checks()


class LRPEpsilon(_LRPFixedParams):
    """LRP-analyzer that uses the LRP-Epsilon rule"""

    def __init__(self, model, epsilon=1e-7, bias=True, *args, **kwargs):
        epsilon = rutils.assert_lrp_epsilon_param(epsilon, self)
        self._epsilon = epsilon

        class EpsilonProxyRule(rrule.EpsilonRule):
            """
            Dummy class inheriting from EpsilonRule
            for passing along the chosen parameters from
            the LRP analyzer class to the decopmosition rules.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, epsilon=epsilon, bias=bias, **kwargs)

        super().__init__(
            model,
            *args,
            rule=EpsilonProxyRule,
            bn_layer_rule=EpsilonProxyRule,
            **kwargs,
        )

        self._do_model_checks()


class LRPWSquare(_LRPFixedParams):
    """LRP-analyzer that uses the DeepTaylor W**2 rule"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model, *args, rule="WSquare", bn_layer_rule="WSquare", **kwargs
        )
        self._do_model_checks()


class LRPFlat(_LRPFixedParams):
    """LRP-analyzer that uses the LRP-Flat rule"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, rule="Flat", bn_layer_rule="Flat", **kwargs)
        self._do_model_checks()


class LRPAlphaBeta(LRP):
    """Base class for LRP AlphaBeta"""

    def __init__(self, model, *args, alpha=None, beta=None, bias=True, **kwargs):
        alpha, beta = rutils.assert_infer_lrp_alpha_beta_param(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta
        self._bias = bias

        class AlphaBetaProxyRule(rrule.AlphaBetaRule):
            """
            Dummy class inheriting from AlphaBetaRule
            for the purpose of passing along the chosen parameters from
            the LRP analyzer class to the decopmosition rules.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, alpha=alpha, beta=beta, bias=bias, **kwargs)

        super().__init__(
            model,
            *args,
            rule=AlphaBetaProxyRule,
            bn_layer_rule=AlphaBetaProxyRule,
            **kwargs,
        )
        self._do_model_checks()

    def _get_state(self):
        state = super()._get_state()
        del state["rule"]
        state.update({"alpha": self._alpha})
        state.update({"beta": self._beta})
        state.update({"bias": self._bias})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        alpha = state.pop("alpha")
        beta = state.pop("beta")
        bias = state.pop("bias")
        state["rule"] = None
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        del kwargs["rule"]
        del kwargs["bn_layer_rule"]
        kwargs.update({"alpha": alpha, "beta": beta, "bias": bias})
        return kwargs


class _LRPAlphaBetaFixedParams(LRPAlphaBeta):
    @classmethod
    def _state_to_kwargs(cls, state):
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        del kwargs["alpha"]
        del kwargs["beta"]
        del kwargs["bias"]
        return kwargs


class LRPAlpha2Beta1(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta rule with a=2,b=1"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, alpha=2, beta=1, bias=True, **kwargs)
        self._do_model_checks()


class LRPAlpha2Beta1IgnoreBias(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta-ignbias rule with a=2,b=1"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, alpha=2, beta=1, bias=False, **kwargs)
        self._do_model_checks()


class LRPAlpha1Beta0(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta rule with a=1,b=0"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, alpha=1, beta=0, bias=True, **kwargs)
        self._do_model_checks()


class LRPAlpha1Beta0IgnoreBias(_LRPAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta-ignbias rule with a=1,b=0"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, alpha=1, beta=0, bias=False, **kwargs)
        self._do_model_checks()


class LRPZPlus(LRPAlpha1Beta0IgnoreBias):
    """LRP-analyzer that uses the LRP-alpha-beta rule with a=1,b=0"""

    # TODO: assert that layer inputs are always >= 0
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._do_model_checks()


class LRPZPlusFast(_LRPFixedParams):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """

    # TODO: assert that layer inputs are always >= 0
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model, *args, rule="ZPlusFast", bn_layer_rule="ZPlusFast", **kwargs
        )
        self._do_model_checks()


class LRPSequentialPresetA(_LRPFixedParams):  # for the lack of a better name
    """Special LRP-configuration for ConvNets"""

    def __init__(
        self,
        model,
        *args,
        epsilon=1e-1,
        bn_layer_rule=rrule.AlphaBetaX2m100Rule,
        **kwargs,
    ):
        class EpsilonProxyRule(rrule.EpsilonRule):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, epsilon=epsilon, bias=True, **kwargs)

        conditional_rules = [
            (ichecks.is_dense_layer, EpsilonProxyRule),
            (ichecks.is_conv_layer, rrule.Alpha1Beta0Rule),
        ]

        super().__init__(
            model, *args, rule=conditional_rules, bn_layer_rule=bn_layer_rule, **kwargs
        )

        self._add_model_check(
            lambda layer: not ichecks.only_relu_activation(layer),
            # TODO: fix. specify. extend.
            (
                "LRPSequentialPresetA is not advised "
                "for networks with non-ReLU activations."
            ),
            check_type="warning",
        )

        self._do_model_checks()


class LRPSequentialPresetB(_LRPFixedParams):
    """Special LRP-configuration for ConvNets"""

    def __init__(
        self,
        model: Model,
        *args,
        epsilon: float = 1e-1,
        bn_layer_rule=rrule.AlphaBetaX2m100Rule,
        **kwargs,
    ):
        class EpsilonProxyRule(rrule.EpsilonRule):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, epsilon=epsilon, bias=True, **kwargs)

        conditional_rules = [
            (ichecks.is_dense_layer, EpsilonProxyRule),
            (ichecks.is_conv_layer, rrule.Alpha2Beta1Rule),
        ]

        super().__init__(
            model, *args, rule=conditional_rules, bn_layer_rule=bn_layer_rule, **kwargs
        )

        # Add and run model checks
        self._add_model_check(
            lambda layer: not ichecks.only_relu_activation(layer),
            # TODO: fix. specify. extend.
            (
                "LRPSequentialPresetB is not advised "
                "for networks with non-ReLU activations."
            ),
            check_type="warning",
        )
        self._do_model_checks()


# TODO: allow to pass input layer identification by index or id.
class LRPSequentialPresetAFlat(LRPSequentialPresetA):
    """Special LRP-configuration for ConvNets"""

    def __init__(self, model, *args, **kwargs):
        # provide functionality for `analyzer.load()` by avoiding multiple kwargs:
        if "input_layer_rule" in kwargs:
            if kwargs["input_layer_rule"] != "Flat":
                raise RuntimeError(
                    "Unexpected input_layer_rule when loading LRPSequentialPresetAFlat."
                )
            kwargs.pop("input_layer_rule")
        super().__init__(model, *args, input_layer_rule="Flat", **kwargs)


# TODO: allow to pass input layer identification by index or id.
class LRPSequentialPresetBFlat(LRPSequentialPresetB):
    """Special LRP-configuration for ConvNets"""

    def __init__(self, model, *args, **kwargs):
        # provide functionality for `analyzer.load()` by avoiding multiple kwargs:
        if "input_layer_rule" in kwargs:
            if kwargs["input_layer_rule"] != "Flat":
                raise RuntimeError(
                    "Unexpected input_layer_rule when loading LRPSequentialPresetBFlat."
                )
            kwargs.pop("input_layer_rule")
        super().__init__(model, *args, input_layer_rule="Flat", **kwargs)


class LRPSequentialPresetBFlatUntilIdx(LRPSequentialPresetBFlat):
    """
    Special LRP-configuration for ConvNets.
    Allows to perform LRP_flat from (including) layer until_layer_idx down until
    the input layer. Weightless layers are ignored when counting the index for now.
    """

    def __init__(self, model, *args, until_layer_idx=None, **kwargs):
        super().__init__(
            model,
            *args,
            until_layer_idx=until_layer_idx,
            until_layer_rule=rrule.FlatRule,
            **kwargs,
        )
