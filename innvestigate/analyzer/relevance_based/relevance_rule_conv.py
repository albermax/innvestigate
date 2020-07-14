# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import zip


###############################################################################
###############################################################################
###############################################################################
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as keras_layers
import numpy as np


from innvestigate import layers as ilayers
from innvestigate import utils as iutils
import innvestigate.utils.keras as kutils
from innvestigate.utils.keras import backend as iK
from innvestigate.utils.keras import graph as kgraph
from . import utils as rutils
from .. import new_base as base


# TODO: differentiate between LRP and DTD rules?
# DTD rules are special cases of LRP rules with additional assumptions
__all__ = [
    #dedicated treatment for special layers


    #general rules
    "ZRule",
    "ZIgnoreBiasRule",

    "EpsilonRule",
    "EpsilonIgnoreBiasRule",

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
    "BoundedRule"
]


class ZRule(base.ReplacementLayer):
    def __init__(self, layer, *args, **kwargs):
        bias = kwargs.pop("bias", True)
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             keep_bias=bias,
                                                             name_template="no_act_%s")
        print(self._layer_wo_act.get_config()["use_bias"])
        super(ZRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Zs = self._layer_wo_act(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0:
                outs = self._neuron_select(outs, neuron_selection)
                Zs = self._neuron_select(Zs, neuron_selection)

        return outs, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, Zs, tape = args
        #last layer
        if reversed_outs is None:
            reversed_outs = Zs

        # Divide incoming relevance by the activations.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])
        # Propagate the relevance to input neurons
        # using the gradient.

        print(self.name, np.shape(reversed_outs), np.shape(ins),np.shape(Zs),np.shape(tmp))
        if len(self.input_shape) > 1:
            raise ValueError("Conv Layers should only have one input!")
        if len(self.layer_next) > 1:
            tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
            #TODO (for all rules) is it correct to add relevances here? should be due to sum conservation?
            ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
        else:
            tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
            ret = keras_layers.Multiply()([ins, tmp2])
        return ret

class ZIgnoreBiasRule(ZRule):
    """
    Basic LRP decomposition rule, ignoring the bias neuron
    """
    def __init__(self, *args, **kwargs):
        super(ZIgnoreBiasRule, self).__init__(*args,
                                              bias=False,
                                              **kwargs)


class EpsilonRule(base.ReplacementLayer):
    """
    Similar to ZRule.
    The only difference is the addition of a numerical stabilizer term
    epsilon to the decomposition function's denominator.
    the sign of epsilon depends on the sign of the output activation
    0 is considered to be positive, ie sign(0) = 1
    """

    def __init__(self, layer, *args, **kwargs):
        self._epsilon = rutils.assert_lrp_epsilon_param(kwargs.pop("epsilon", 1e-7), self)
        bias = kwargs.pop("bias", True)
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             keep_bias=bias,
                                                             name_template="no_act_%s")
        super(EpsilonRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Zs = self._layer_wo_act(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0:
                outs = self._neuron_select(outs, neuron_selection)
                Zs = self._neuron_select(Zs, neuron_selection)

        return outs, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, Zs, tape = args

        if reversed_outs is None:
            reversed_outs = Zs

        # The epsilon rule aligns epsilon with the (extended) sign: 0 is considered to be positive
        prepare_div = keras_layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * self._epsilon)
        # Divide incoming relevance by the activations.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, prepare_div(Zs)]) for r in reversed_outs]
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, prepare_div(Zs)])
        # Propagate the relevance to input neurons
        # using the gradient.
        if len(self.input_shape) > 1:
            raise ValueError("Conv Layers should only have one input!")
        if len(self.layer_next) > 1:
            tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
            ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
        else:
            tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
            ret = keras_layers.Multiply()([ins, tmp2])
        return ret

class EpsilonIgnoreBiasRule(EpsilonRule):
    """Same as EpsilonRule but ignores the bias."""
    def __init__(self, *args, **kwargs):
        super(EpsilonIgnoreBiasRule, self).__init__(*args,
                                                    bias=False,
                                                    **kwargs)

class WSquareRule(base.ReplacementLayer):
    """W**2 rule from Deep Taylor Decomposition"""

    def __init__(self, layer, *args, **kwargs):
        copy_weights = kwargs.pop("copy_weights", True)
        # W-square rule works with squared weights and no biases.
        if copy_weights:
            weights = layer.get_weights()
        else:
            weights = layer.weights
        if layer.use_bias:
            weights = weights[:-1]
        weights = [x**2 for x in weights]

        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_%s")
        super(WSquareRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Ys = self._layer_wo_act_b(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0:
                outs = self._neuron_select(outs, neuron_selection)
                Ys = self._neuron_select(Ys, neuron_selection)

        return outs, Ys, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, Ys, tape = args
        # Compute the sum of the weights.
        ones = ilayers.OnesLike()(ins)
        Zs = self._layer_wo_act_b(ones)

        if reversed_outs is None:
            reversed_outs = Zs

        # Weight the incoming relevance.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])

        # Redistribute the relevances along the gradient.
        if len(self.input_shape) > 1:
            raise ValueError("Conv Layers should only have one input!")
        if len(self.layer_next) > 1:
            ret = keras_layers.Add()([tape.gradient(Ys, ins, output_gradients=t) for t in tmp])
        else:
            ret = tape.gradient(Ys, ins, output_gradients=tmp)
        return ret

class FlatRule(WSquareRule):
    """Same as W**2 rule but sets all weights to ones."""

    def __init__(self, layer, *args, **kwargs):
        copy_weights = kwargs.pop("copy_weights", True)
        # The flat rule works with weights equal to one and
        # no biases.
        if copy_weights:
            weights = layer.get_weights()
            if layer.use_bias:
                weights = weights[:-1]
            weights = [np.ones_like(x) for x in weights]
        else:
            weights = layer.weights
            if layer.use_bias:
                weights = weights[:-1]
            weights = [K.ones_like(x) for x in weights]

        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_%s")
        super(WSquareRule, self).__init__(layer, *args, **kwargs)

class AlphaBetaRule(base.ReplacementLayer):
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


    def __init__(self,
                 layer,
                 *args,
                 alpha=None,
                 beta=None,
                 bias=True,
                 copy_weights=False,
                 **kwargs):
        alpha, beta = rutils.assert_infer_lrp_alpha_beta_param(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
        if copy_weights:
            weights = layer.get_weights()
            if not bias and layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if not bias and layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * iK.to_floatx(x > 0) for x in weights]
            negative_weights = [x * iK.to_floatx(x < 0) for x in weights]

        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s")
        super(AlphaBetaRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection):
        keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
        keep_negatives = keras_layers.Lambda(lambda x: x * K.cast(K.less(x, 0), K.floatx()))
        ins_pos = keep_positives(ins)
        ins_neg = keep_negatives(ins)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            tape.watch(ins_pos)
            tape.watch(ins_neg)
            outs = self.layer_func(ins)
            Zs_pos = self._layer_wo_act_positive(ins_pos)
            Zs_neg = self._layer_wo_act_negative(ins_neg)
            Zs_pos_n = self._layer_wo_act_negative(ins_pos)
            Zs_neg_p = self._layer_wo_act_positive(ins_neg)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0:
                outs = self._neuron_select(outs, neuron_selection)
                Zs_pos = self._neuron_select(Zs_pos, neuron_selection)
                Zs_neg = self._neuron_select(Zs_neg, neuron_selection)
                Zs_pos_n = self._neuron_select(Zs_pos_n, neuron_selection)
                Zs_neg_p = self._neuron_select(Zs_neg_p, neuron_selection)

        return outs, ins_pos, ins_neg, Zs_pos, Zs_neg, Zs_pos_n, Zs_neg_p, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, ins_pos, ins_neg, Zs_pos, Zs_neg, Zs_pos_n, Zs_neg_p, tape = args
        # this method is correct, but wasteful
        times_alpha = keras_layers.Lambda(lambda x: x * self._alpha)
        times_beta = keras_layers.Lambda(lambda x: x * self._beta)

        def f(i1, i2, z1, z2, rev):
            Zs = [keras_layers.Add()([a, b]) for a, b in zip(z1, z2)]
            # Divide incoming relevance by the activations.
            if rev is None:
                rev = Zs
            if len(self.layer_next) > 1:
                tmp = [ilayers.SafeDivide()([r, Zs]) for r in rev]
            else:
                tmp = ilayers.SafeDivide()([rev, Zs])
            # Propagate the relevance to the input neurons
            # using the gradient
            if len(self.input_shape) > 1:
                raise ValueError("Conv Layers should only have one input!")
            if len(self.layer_next) > 1:
                tmp1 = [tape.gradient(z1, i1, output_gradients=tmp) for t in tmp]
                tmp2 = [tape.gradient(z2, i2, output_gradients=tmp) for t in tmp]
                # Re-weight relevance with the input values.
                tmp_1 = [keras_layers.Multiply()([i1, tmp1]) for t in tmp1]
                tmp_2 = [keras_layers.Multiply()([i2, tmp2]) for t in tmp2]
                # combine
                combined = [keras_layers.Add()([a, b]) for a, b in zip(tmp_1, tmp_2)]
            else:
                tmp1 = tape.gradient(z1, i1, output_gradients=tmp)
                tmp2 = tape.gradient(z2, i2, output_gradients=tmp)
                # Re-weight relevance with the input values.

                tmp_1 = keras_layers.Multiply()([i1, tmp1])
                tmp_2 = keras_layers.Multiply()([i2, tmp2])
                # combine
                combined = keras_layers.Add()([tmp_1, tmp_2])
            return combined

        # xpos*wpos + xneg*wneg
        activator_relevances = f(ins_pos, ins_neg, Zs_pos,Zs_neg,reversed_outs)

        if self._beta:  # only compute beta-weighted contributions of beta is not zero
            # xpos*wneg + xneg*wpos
            inhibitor_relevances = f(ins_pos, ins_neg, Zs_pos_n,Zs_neg_p,reversed_outs)
            if len(self.layer_next) > 1:
                sub = [keras_layers.Subtract()([times_alpha(a), times_beta(b)]) for a, b in zip(activator_relevances, inhibitor_relevances)]
                ret = keras_layers.Add()(sub)
            else:
                ret = keras_layers.Subtract()([times_alpha(activator_relevances), times_beta(inhibitor_relevances)])
            return ret
        else:
            if len(self.layer_next) > 1:
                ret = keras_layers.Add()(activator_relevances)
            else:
                ret = activator_relevances
            return ret

class AlphaBetaIgnoreBiasRule(AlphaBetaRule):
    """Same as AlphaBetaRule but ignores biases."""
    def __init__(self, *args, **kwargs):
        super(AlphaBetaIgnoreBiasRule, self).__init__(*args,
                                                      bias=False,
                                                      **kwargs)

class Alpha2Beta1Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1"""
    def __init__(self, *args, **kwargs):
        super(Alpha2Beta1Rule, self).__init__(*args,
                                              alpha=2,
                                              beta=1,
                                              bias=True,
                                              **kwargs)

class Alpha2Beta1IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1 and ignores biases"""
    def __init__(self, *args, **kwargs):
        super(Alpha2Beta1IgnoreBiasRule, self).__init__(*args,
                                                        alpha=2,
                                                        beta=1,
                                                        bias=False,
                                                        **kwargs)

class Alpha1Beta0Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0"""
    def __init__(self, *args, **kwargs):
        super(Alpha1Beta0Rule, self).__init__(*args,
                                              alpha=1,
                                              beta=0,
                                              bias=True,
                                              **kwargs)

class Alpha1Beta0IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0 and ignores biases"""
    def __init__(self, *args, **kwargs):
        super(Alpha1Beta0IgnoreBiasRule, self).__init__(*args,
                                                        alpha=1,
                                                        beta=0,
                                                        bias=False,
                                                        **kwargs)

#TODO: tf2.0
class AlphaBetaXRule(kgraph.ReverseMappingBase):
    """
    AlphaBeta advanced as proposed by Alexander Binder.
    """

    def __init__(self,
                 layer,
                 state,
                 alpha=(0.5, 0.5),
                 beta=(0.5, 0.5),
                 bias=True,
                 copy_weights=False):
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
            positive_weights = [x * iK.to_floatx(x > 0) for x in weights]
            negative_weights = [x * iK.to_floatx(x < 0) for x in weights]

        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        #this method is correct, but wasteful
        grad = ilayers.GradientWRT(len(Xs))
        times_alpha0 = keras_layers.Lambda(lambda x: x * self._alpha[0])
        times_alpha1 = keras_layers.Lambda(lambda x: x * self._alpha[1])
        times_beta0 = keras_layers.Lambda(lambda x: x * self._beta[0])
        times_beta1 = keras_layers.Lambda(lambda x: x * self._beta[1])
        keep_positives = keras_layers.Lambda(
            lambda x: x * K.cast(K.greater(x,0), K.floatx()))
        keep_negatives = keras_layers.Lambda(
            lambda x: x * K.cast(K.less(x,0), K.floatx()))

        def f(layer, X):
            Zs = kutils.apply(layer, X)
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([a, b])
                    for a, b in zip(Rs, Zs)]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp = iutils.to_list(grad(X+Zs+tmp))
            # Re-weight relevance with the input values.
            tmp = [keras_layers.Multiply()([a, b])
                    for a, b in zip(X, tmp)]
            return tmp

        # Distinguish postive and negative inputs.
        Xs_pos = kutils.apply(keep_positives, Xs)
        Xs_neg = kutils.apply(keep_negatives, Xs)

        # xpos*wpos
        r_pp = f(self._layer_wo_act_positive, Xs_pos)
        # xneg*wneg
        r_nn = f(self._layer_wo_act_negative, Xs_neg)
        # a0 * r_pp + a1 * r_nn
        r_pos = [keras_layers.Add()([times_alpha0(pp), times_beta1(nn)])
                 for pp, nn in zip(r_pp, r_nn)]

        # xpos*wneg
        r_pn = f(self._layer_wo_act_negative, Xs_pos)
        # xneg*wpos
        r_np = f(self._layer_wo_act_positive, Xs_neg)
        # b0 * r_pn + b1 * r_np
        r_neg = [keras_layers.Add()([times_beta0(pn), times_beta1(np)])
                 for pn, np in zip(r_pn, r_np)]

        return [keras_layers.Subtract()([a, b]) for a, b in zip(r_pos, r_neg)]

#TODO: tf2.0
class AlphaBetaX1000Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1000Rule, self).__init__(*args,
                                                 alpha=(1, 0),
                                                 beta=(0, 0),
                                                 bias=True,
                                                 **kwargs)

#TODO: tf2.0
class AlphaBetaX1010Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1010Rule, self).__init__(*args,
                                                 alpha=(1, 0),
                                                 beta=(0, -1),
                                                 bias=True,
                                                 **kwargs)

#TODO: tf2.0
class AlphaBetaX1001Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1001Rule, self).__init__(*args,
                                                 alpha=(1, 1),
                                                 beta=(0, 0),
                                                 bias=True,
                                                 **kwargs)

#TODO: tf2.0
class AlphaBetaX2m100Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX2m100Rule, self).__init__(*args,
                                                  alpha=(2, 0),
                                                  beta=(1, 0),
                                                  bias=True,
                                                  **kwargs)

#TODO: tf2.0
class BoundedRule(kgraph.ReverseMappingBase):
    """Z_B rule from the Deep Taylor Decomposition"""
    # TODO: this only works for relu networks, needs to be extended.
    # TODO: check
    def __init__(self, layer, state, low=-1, high=1, copy_weights=False):
        self._low = low
        self._high = high

        # This rule works with three variants of the layer, all without biases.
        # One is the original form and two with only the positive or
        # negative weights.
        if copy_weights:
            weights = layer.get_weights()
            if layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * iK.to_floatx(x > 0) for x in weights]
            negative_weights = [x * iK.to_floatx(x < 0) for x in weights]

        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_%s")
        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s")

    # TODO: clean up this implementation and add more documentation
    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        to_low = keras_layers.Lambda(lambda x: x * 0 + self._low)
        to_high = keras_layers.Lambda(lambda x: x * 0 + self._high)

        low = [to_low(x) for x in Xs]
        high = [to_high(x) for x in Xs]

        # Get values for the division.
        A = kutils.apply(self._layer_wo_act, Xs)
        B = kutils.apply(self._layer_wo_act_positive, low)
        C = kutils.apply(self._layer_wo_act_negative, high)
        Zs = [keras_layers.Subtract()([a, keras_layers.Add()([b, c])])
              for a, b, c in zip(A, B, C)]

        # Divide relevances with the value.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        tmpA = iutils.to_list(grad(Xs+A+tmp))
        tmpB = iutils.to_list(grad(low+B+tmp))
        tmpC = iutils.to_list(grad(high+C+tmp))

        tmpA = [keras_layers.Multiply()([a, b]) for a, b in zip(Xs, tmpA)]
        tmpB = [keras_layers.Multiply()([a, b]) for a, b in zip(low, tmpB)]
        tmpC = [keras_layers.Multiply()([a, b]) for a, b in zip(high, tmpC)]

        tmp = [keras_layers.Subtract()([a, keras_layers.Add()([b, c])])
               for a, b, c in zip(tmpA, tmpB, tmpC)]

        return tmp


#TODO: tf2.0
class ZPlusRule(Alpha1Beta0IgnoreBiasRule):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0, which assumes inputs x >= 0
    and ignores the bias.
    CAUTION! Results differ from Alpha=1, Beta=0
    if inputs are not strictly >= 0
    """
    #TODO: assert that layer inputs are always >= 0
    def __init__(self, *args, **kwargs):
        super(ZPlusRule, self).__init__(*args, **kwargs)


#TODO: tf2.0
class ZPlusFastRule(kgraph.ReverseMappingBase):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """

    def __init__(self, layer, state, copy_weights=False):
        # The z-plus rule only works with positive weights and
        # no biases.
        #TODO: assert that layer inputs are always >= 0
        if copy_weights:
            weights = layer.get_weights()
            if layer.use_bias:
                weights = weights[:-1]
            weights = [x * (x > 0) for x in weights]
        else:
            weights = layer.weights
            if layer.use_bias:
                weights = weights[:-1]
            weights = [x * iK.to_floatx(x > 0) for x in weights]

        self._layer_wo_act_b_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_positive_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        #TODO: assert all inputs are positive, instead of only keeping the positives.
        #keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x,0), K.floatx()))
        #Xs = kutils.apply(keep_positives, Xs)

        # Get activations.
        Zs = kutils.apply(self._layer_wo_act_b_positive, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras_layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]

#TODO: gamma-Rule tf2.0