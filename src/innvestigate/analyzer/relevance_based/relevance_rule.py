from __future__ import annotations

import numpy as np
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers

import innvestigate.analyzer.relevance_based.utils as rutils
import innvestigate.backend as ibackend
import innvestigate.backend.graph as igraph
import innvestigate.layers as ilayers
from innvestigate.backend.types import Layer, OptionalList, Tensor

# TODO: differentiate between LRP and DTD rules?
# DTD rules are special cases of LRP rules with additional assumptions
__all__ = [
    # dedicated treatment for special layers
    # general rules
    "ZRule",
    "EpsilonRule",
    "WSquareRule",
    "FlatRule",
    "AlphaBetaRule",
    "AlphaBetaIgnoreBiasRule",
    "Alpha2Beta1Rule",
    "Alpha2Beta1IgnoreBiasRule",
    "Alpha1Beta0Rule",
    "Alpha1Beta0IgnoreBiasRule",
    "AlphaBetaXRule",
    "AlphaBetaX1000Rule",
    "AlphaBetaX1010Rule",
    "AlphaBetaX1001Rule",
    "AlphaBetaX2m100Rule",
    "ZPlusRule",
    "ZPlusFastRule",
    "BoundedRule",
]


class ZRule(igraph.ReverseMappingBase):
    """
    Basic LRP decomposition rule (for layers with weight kernels),
    which considers the bias a constant input neuron.
    """

    def __init__(self, layer: Layer, _state, bias: bool = True) -> None:
        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s"
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state,
    ) -> list[Tensor]:

        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]


class EpsilonRule(igraph.ReverseMappingBase):
    """
    Similar to ZRule.
    The only difference is the addition of a numerical stabilizer term
    epsilon to the decomposition function's denominator.
    the sign of epsilon depends on the sign of the output activation
    0 is considered to be positive, ie sign(0) = 1
    """

    def __init__(self, layer: Layer, _state, epsilon=1e-7, bias: bool = True):
        self._epsilon = rutils.assert_lrp_epsilon_param(epsilon, self)
        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s"
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ):
        # The epsilon rule aligns epsilon with the (extended) sign:
        # 0 is considered to be positive
        prepare_div = klayers.Lambda(
            lambda x: x
            + (kbackend.cast(kbackend.greater_equal(x, 0), kbackend.floatx()) * 2 - 1)
            * self._epsilon
        )

        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act, Xs)

        # Divide incoming relevance by the activations.
        tmp = [a / prepare_div(b) for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]


class WSquareRule(igraph.ReverseMappingBase):
    """W**2 rule from Deep Taylor Decomposition"""

    def __init__(self, layer: Layer, _state, copy_weights=False) -> None:
        # W-square rule works with squared weights and no biases.
        if copy_weights:
            weights = layer.get_weights()
        else:
            weights = layer.weights
        if getattr(layer, "use_bias", False):
            weights = weights[:-1]
        weights = [x**2 for x in weights]

        self._layer_wo_act_b = igraph.copy_layer_wo_activation(
            layer, keep_bias=False, weights=weights, name_template="reversed_kernel_%s"
        )

    def apply(
        self,
        Xs: list[Tensor],
        Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ) -> list[Tensor]:
        # Create dummy forward path to take the derivative below.
        Ys = ibackend.apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = [self._layer_wo_act_b(X) for X in ones]
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b]) for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        grads = ibackend.gradients(Xs, Ys, tmp)
        return grads


class FlatRule(WSquareRule):
    """Same as W**2 rule but sets all weights to ones."""

    def __init__(self, layer: Layer, _state, copy_weights: bool = False) -> None:
        # The flat rule works with weights equal to one and
        # no biases.
        if copy_weights:
            weights = layer.get_weights()
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [np.ones_like(x) for x in weights]
        else:
            weights = layer.weights
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [kbackend.ones_like(x) for x in weights]

        self._layer_wo_act_b = igraph.copy_layer_wo_activation(
            layer, keep_bias=False, weights=weights, name_template="reversed_kernel_%s"
        )


class AlphaBetaRule(igraph.ReverseMappingBase):
    """
    This decomposition rule handles the positive forward
    activations (x*w > 0) and negative forward activations
    (w * x < 0) independently, reducing the risk of zero
    divisions considerably. In fact, the only case where
    divisions by zero can happen is if there are either
    no positive or no negative parts to the activation
    at all.
    Corresponding parameterization of this rule implement
    methods such as Excitation Backpropagation with
    alpha=1, beta=0
    s.t.
    alpha - beta = 1 (after current param. scheme.)
    and
    alpha > 1
    beta > 0
    """

    def __init__(
        self,
        layer: Layer,
        _state,
        alpha=None,
        beta=None,
        bias: bool = True,
        copy_weights=False,
    ) -> None:
        alpha, beta = rutils.assert_infer_lrp_alpha_beta_param(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
        if copy_weights:
            weights = layer.get_weights()
            if not bias and getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if not bias and getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * ibackend.cast_to_floatx(x > 0) for x in weights]
            negative_weights = [x * ibackend.cast_to_floatx(x < 0) for x in weights]

        self._layer_wo_act_positive = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s",
        )
        self._layer_wo_act_negative = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s",
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ):
        # this method is correct, but wasteful
        times_alpha = klayers.Lambda(lambda x: x * self._alpha)
        times_beta = klayers.Lambda(lambda x: x * self._beta)
        keep_positives = klayers.Lambda(
            lambda x: x * kbackend.cast(kbackend.greater(x, 0), kbackend.floatx())
        )
        keep_negatives = klayers.Lambda(
            lambda x: x * kbackend.cast(kbackend.less(x, 0), kbackend.floatx())
        )

        def fn_tmp(
            layer1: Layer,
            layer2: Layer,
            X1: OptionalList[Tensor],
            X2: OptionalList[Tensor],
        ):
            # Get activations of full positive or negative part.
            Z1 = ibackend.apply(layer1, X1)
            Z2 = ibackend.apply(layer2, X2)
            Zs = [klayers.Add()([a, b]) for a, b in zip(Z1, Z2)]
            # Divide incoming relevance by the activations.
            tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
            # Propagate the relevance to the input neurons
            # using the gradient
            grads1 = ibackend.gradients(X1, Z1, tmp)
            grads2 = ibackend.gradients(X2, Z2, tmp)
            # Re-weight relevance with the input values.
            tmp1 = [klayers.Multiply()([a, b]) for a, b in zip(X1, grads1)]
            tmp2 = [klayers.Multiply()([a, b]) for a, b in zip(X2, grads2)]
            # combine and return
            return [klayers.Add()([a, b]) for a, b in zip(tmp1, tmp2)]

        # Distinguish postive and negative inputs.
        Xs_pos = ibackend.apply(keep_positives, Xs)
        Xs_neg = ibackend.apply(keep_negatives, Xs)
        # xpos*wpos + xneg*wneg
        activator_relevances = fn_tmp(
            self._layer_wo_act_positive, self._layer_wo_act_negative, Xs_pos, Xs_neg
        )

        if self._beta:  # only compute beta-weighted contributions of beta is not zero
            # xpos*wneg + xneg*wpos
            inhibitor_relevances = fn_tmp(
                self._layer_wo_act_negative, self._layer_wo_act_positive, Xs_pos, Xs_neg
            )
            return [
                klayers.Subtract()([times_alpha(a), times_beta(b)])
                for a, b in zip(activator_relevances, inhibitor_relevances)
            ]
        return activator_relevances


class AlphaBetaIgnoreBiasRule(AlphaBetaRule):
    """Same as AlphaBetaRule but ignores biases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias=False, **kwargs)


class Alpha2Beta1Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=2, beta=1, bias=True, **kwargs)


class Alpha2Beta1IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1 and ignores biases"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=2, beta=1, bias=False, **kwargs)


class Alpha1Beta0Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=1, beta=0, bias=True, **kwargs)


class Alpha1Beta0IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0 and ignores biases"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=1, beta=0, bias=False, **kwargs)


class AlphaBetaXRule(igraph.ReverseMappingBase):
    """
    AlphaBeta advanced as proposed by Alexander Binder.
    """

    def __init__(
        self,
        layer: Layer,
        _state,
        alpha: tuple[float, float] = (0.5, 0.5),
        beta: tuple[float, float] = (0.5, 0.5),
        bias: bool = True,
        copy_weights: bool = False,
    ) -> None:
        self._alpha = alpha
        self._beta = beta

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
        if copy_weights:
            weights = layer.get_weights()
            if not bias and getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if not bias and getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * ibackend.cast_to_floatx(x > 0) for x in weights]
            negative_weights = [x * ibackend.cast_to_floatx(x < 0) for x in weights]

        self._layer_wo_act_positive = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s",
        )
        self._layer_wo_act_negative = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s",
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ):
        # this method is correct, but wasteful
        times_alpha0 = klayers.Lambda(lambda x: x * self._alpha[0])
        # times_alpha1 = klayers.Lambda(lambda x: x * self._alpha[1]) # unused
        times_beta0 = klayers.Lambda(lambda x: x * self._beta[0])
        times_beta1 = klayers.Lambda(lambda x: x * self._beta[1])
        keep_positives = klayers.Lambda(
            lambda x: x * kbackend.cast(kbackend.greater(x, 0), kbackend.floatx())
        )
        keep_negatives = klayers.Lambda(
            lambda x: x * kbackend.cast(kbackend.less(x, 0), kbackend.floatx())
        )

        def fn_tmp(layer: Layer, Xs: OptionalList[Tensor]):
            Zs = ibackend.apply(layer, Xs)
            # Divide incoming relevance by the activations.
            tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
            # Propagate the relevance to the input neurons
            # using the gradient
            grads = ibackend.gradients(Xs, Zs, tmp)
            # Re-weight relevance with the input values.
            return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]

        # Distinguish postive and negative inputs.
        Xs_pos = ibackend.apply(keep_positives, Xs)
        Xs_neg = ibackend.apply(keep_negatives, Xs)

        # xpos*wpos
        r_pp = fn_tmp(self._layer_wo_act_positive, Xs_pos)
        # xneg*wneg
        r_nn = fn_tmp(self._layer_wo_act_negative, Xs_neg)
        # a0 * r_pp + a1 * r_nn
        r_pos = [
            klayers.Add()([times_alpha0(pp), times_beta1(nn)])
            for pp, nn in zip(r_pp, r_nn)
        ]

        # xpos*wneg
        r_pn = fn_tmp(self._layer_wo_act_negative, Xs_pos)
        # xneg*wpos
        r_np = fn_tmp(self._layer_wo_act_positive, Xs_neg)
        # b0 * r_pn + b1 * r_np
        r_neg = [
            klayers.Add()([times_beta0(pn), times_beta1(np)])
            for pn, np in zip(r_pn, r_np)
        ]

        return [klayers.Subtract()([a, b]) for a, b in zip(r_pos, r_neg)]


class AlphaBetaX1000Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(1, 0), beta=(0, 0), bias=True, **kwargs)


class AlphaBetaX1010Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(1, 0), beta=(0, -1), bias=True, **kwargs)


class AlphaBetaX1001Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(1, 1), beta=(0, 0), bias=True, **kwargs)


class AlphaBetaX2m100Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(2, 0), beta=(1, 0), bias=True, **kwargs)


class BoundedRule(igraph.ReverseMappingBase):
    """Z_B rule from the Deep Taylor Decomposition"""

    # TODO: this only works for relu networks, needs to be extended.
    # TODO: check
    def __init__(
        self, layer: Layer, _state, low=-1, high=1, copy_weights: bool = False
    ) -> None:
        self._low = low
        self._high = high

        # This rule works with three variants of the layer, all without biases.
        # One is the original form and two with only the positive or
        # negative weights.
        if copy_weights:
            weights = layer.get_weights()
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * ibackend.cast_to_floatx(x > 0) for x in weights]
            negative_weights = [x * ibackend.cast_to_floatx(x < 0) for x in weights]

        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s"
        )
        self._layer_wo_act_positive = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s",
        )
        self._layer_wo_act_negative = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s",
        )

    # TODO: clean up this implementation and add more documentation
    def apply(self, Xs, _Ys, Rs, reverse_state: dict):
        to_low = klayers.Lambda(lambda x: x * 0 + self._low)
        to_high = klayers.Lambda(lambda x: x * 0 + self._high)

        low = [to_low(x) for x in Xs]
        high = [to_high(x) for x in Xs]

        # Get values for the division.
        A = ibackend.apply(self._layer_wo_act, Xs)
        B = ibackend.apply(self._layer_wo_act_positive, low)
        C = ibackend.apply(self._layer_wo_act_negative, high)
        Zs = [
            klayers.Subtract()([a, klayers.Add()([b, c])]) for a, b, c in zip(A, B, C)
        ]

        # Divide relevances with the value.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        grads_a = ibackend.gradients(Xs, A, tmp)
        grads_b = ibackend.gradients(low, B, tmp)
        grads_c = ibackend.gradients(high, C, tmp)

        tmp_a = [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads_a)]
        tmp_b = [klayers.Multiply()([a, b]) for a, b in zip(low, grads_b)]
        tmp_c = [klayers.Multiply()([a, b]) for a, b in zip(high, grads_c)]

        ret = [
            klayers.Subtract()([a, klayers.Add()([b, c])])
            for a, b, c in zip(tmp_a, tmp_b, tmp_c)
        ]

        return ret


class ZPlusRule(Alpha1Beta0IgnoreBiasRule):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0, which assumes inputs x >= 0
    and ignores the bias.
    CAUTION! Results differ from Alpha=1, Beta=0
    if inputs are not strictly >= 0
    """

    # TODO: assert that layer inputs are always >= 0


class ZPlusFastRule(igraph.ReverseMappingBase):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """

    def __init__(self, layer: Layer, _state, copy_weights=False):
        # The z-plus rule only works with positive weights and
        # no biases.
        # TODO: assert that layer inputs are always >= 0
        if copy_weights:
            weights = layer.get_weights()
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [x * (x > 0) for x in weights]
        else:
            weights = layer.weights
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [x * ibackend.cast_to_floatx(x > 0) for x in weights]

        self._layer_wo_act_b_positive = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_positive_%s",
        )

    def apply(self, Xs, _Ys, Rs, reverse_state: dict):
        # TODO: assert all inputs are positive, instead of only keeping the positives.
        # keep_positives = klayers.Lambda(
        #     lambda x: x * kbackend.cast(kbackend.greater(x, 0), kbackend.floatx())
        # )
        # Xs = ibackend.apply(keep_positives, Xs)

        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act_b_positive, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]
