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

from ... import layers as ilayers
from ... import utils as iutils
from ...utils import keras as kutils
from ...utils.keras import backend as iK
from ...utils.keras import graph as kgraph
from . import utils as rutils
from .. import reverse_map as reverse_map


# TODO: differentiate between LRP and DTD rules?
# DTD rules are special cases of LRP rules with additional assumptions
__all__ = [
    #dedicated treatment for special layers

    "BatchNormalizationReverseRule",
    "AddReverseRule",
    "AveragePoolingReverseRule",

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

class ZRule(reverse_map.ReplacementLayer):
    def __init__(self, layer, *args, **kwargs):
        bias = kwargs.pop("bias", True)
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             keep_bias=bias,
                                                             name_template="no_act_%s")
        #print(self._layer_wo_act.get_config()["use_bias"])
        super(ZRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Zs = self._layer_wo_act(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)

        return outs, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, Zs, tape = args
        #last layer
        if reversed_outs is None:
            reversed_outs = Zs

        # Divide incoming relevance by the activations.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
            ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])
            # Propagate the relevance to input neurons
            # using the gradient.
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


class EpsilonRule(reverse_map.ReplacementLayer):
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

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Zs = self._layer_wo_act(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)

        return outs, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, Zs, tape = args

        if reversed_outs is None:
            reversed_outs = Zs

        # The epsilon rule aligns epsilon with the (extended) sign: 0 is considered to be positive
        prepare_div = keras_layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * self._epsilon)
        # Divide incoming relevance by the activations.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, prepare_div(Zs)]) for r in reversed_outs]
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
            ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, prepare_div(Zs)])
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
            ret = keras_layers.Multiply()([ins, tmp2])
        return ret

class EpsilonIgnoreBiasRule(EpsilonRule):
    """Same as EpsilonRule but ignores the bias."""
    def __init__(self, *args, **kwargs):
        super(EpsilonIgnoreBiasRule, self).__init__(*args,
                                                    bias=False,
                                                    **kwargs)

class WSquareRule(reverse_map.ReplacementLayer):
    """W**2 rule from Deep Taylor Decomposition"""

    def __init__(self, layer, *args, **kwargs):
        copy_weights = kwargs.pop("copy_weights", False)
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

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            ones = ilayers.OnesLike()(ins)
            outs = self.layer_func(ins)
            Ys = self._layer_wo_act_b(ins)
            Zs = self._layer_wo_act_b(ones)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Ys = self._neuron_sel_and_head_map(Ys, neuron_selection, r_init)
                # Compute the sum of the weights.
                Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)

        return outs, Ys, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, Ys, Zs, tape = args

        if reversed_outs is None:
            reversed_outs = outs

        # Weight the incoming relevance.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
            # Redistribute the relevances along the gradient.
            ret = keras_layers.Add()([tape.gradient(Ys, ins, output_gradients=t) for t in tmp])
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])
            # Redistribute the relevances along the gradient.
            ret = tape.gradient(Ys, ins, output_gradients=tmp)

        return ret

class FlatRule(WSquareRule):
    """Same as W**2 rule but sets all weights to ones."""

    def __init__(self, layer, *args, **kwargs):
        copy_weights = kwargs.pop("copy_weights", False)
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

class AlphaBetaRule(reverse_map.ReplacementLayer):
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

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
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
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs_pos = self._neuron_sel_and_head_map(Zs_pos, neuron_selection, r_init)
                Zs_neg = self._neuron_sel_and_head_map(Zs_neg, neuron_selection, r_init)
                Zs_pos_n = self._neuron_sel_and_head_map(Zs_pos_n, neuron_selection, r_init)
                Zs_neg_p = self._neuron_sel_and_head_map(Zs_neg_p, neuron_selection, r_init)

        return outs, ins_pos, ins_neg, Zs_pos, Zs_neg, Zs_pos_n, Zs_neg_p, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, ins_pos, ins_neg, Zs_pos, Zs_neg, Zs_pos_n, Zs_neg_p, tape = args
        # this method is correct, but wasteful
        times_alpha = keras_layers.Lambda(lambda x: x * self._alpha)
        times_beta = keras_layers.Lambda(lambda x: x * self._beta)

        def f(i1, i2, z1, z2, rev):
            if rev is None:
                rev = outs

            Zs = keras_layers.Add()([z1, z2])

            # Divide incoming relevance by the activations.
            if len(self.layer_next) > 1:
                tmp = [ilayers.SafeDivide()([r, Zs]) for r in rev]
                # Propagate the relevance to the input neurons
                # using the gradient
                tmp1 = [tape.gradient(z1, i1, output_gradients=t) for t in tmp]
                tmp2 = [tape.gradient(z2, i2, output_gradients=t) for t in tmp]
                # Re-weight relevance with the input values.
                tmp_1 = [keras_layers.Multiply()([i1, t]) for t in tmp1]
                tmp_2 = [keras_layers.Multiply()([i2, t]) for t in tmp2]
                # combine
                combined = [keras_layers.Add()([a, b]) for a, b in zip(tmp_1, tmp_2)]
            else:
                tmp = ilayers.SafeDivide()([rev, Zs])
                # Propagate the relevance to the input neurons
                # using the gradient
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

class AlphaBetaXRule(reverse_map.ReplacementLayer):
    """
    AlphaBeta advanced as proposed by Alexander Binder.
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
        super(AlphaBetaXRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
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
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs_pos = self._neuron_sel_and_head_map(Zs_pos, neuron_selection, r_init)
                Zs_neg = self._neuron_sel_and_head_map(Zs_neg, neuron_selection, r_init)
                Zs_pos_n = self._neuron_sel_and_head_map(Zs_pos_n, neuron_selection, r_init)
                Zs_neg_p = self._neuron_sel_and_head_map(Zs_neg_p, neuron_selection, r_init)

        return outs, ins_pos, ins_neg, Zs_pos, Zs_neg, Zs_pos_n, Zs_neg_p, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, ins_pos, ins_neg, Zs_pos, Zs_neg, Zs_pos_n, Zs_neg_p, tape = args
        # this method is correct, but wasteful
        times_alpha0 = keras_layers.Lambda(lambda x: x * self._alpha[0])
        times_alpha1 = keras_layers.Lambda(lambda x: x * self._alpha[1])
        times_beta0 = keras_layers.Lambda(lambda x: x * self._beta[0])
        times_beta1 = keras_layers.Lambda(lambda x: x * self._beta[1])

        def f(Xs, Zs, rev):
            # Divide incoming relevance by the activations.
            if rev is None:
                rev = outs
            if len(self.layer_next) > 1:
                tmp = [ilayers.SafeDivide()([r, Zs]) for r in rev]
                # Propagate the relevance to the input neurons
                # using the gradient
                tmp1 = [tape.gradient(Zs, Xs, output_gradients=t) for t in tmp]
                # Re-weight relevance with the input values.
                tmp_1 = [keras_layers.Multiply()([Xs, t]) for t in tmp1]
            else:
                tmp = ilayers.SafeDivide()([rev, Zs])
                # Propagate the relevance to the input neurons
                # using the gradient
                tmp1 = tape.gradient(Zs, Xs, output_gradients=tmp)
                # Re-weight relevance with the input values.
                tmp_1 = keras_layers.Multiply()([Xs, tmp1])

            return tmp_1

        # xpos*wpos
        r_pp = f(ins_pos, Zs_pos, reversed_outs)
        # xneg*wneg
        r_nn = f(ins_neg, Zs_neg, reversed_outs)
        # a0 * r_pp + a1 * r_nn
        if len(self.layer_next) > 1:
            r_pos = [keras_layers.Add()([times_alpha0(pp), times_alpha1(nn)]) for pp, nn in zip(r_pp, r_nn)]
        else:
            r_pos = keras_layers.Add()([times_alpha0(r_pp), times_alpha1(r_nn)])

        # xpos*wneg
        r_pn = f(ins_pos, Zs_pos_n, reversed_outs)
        # xneg*wpos
        r_np = f(ins_neg, Zs_neg_p, reversed_outs)
        # b0 * r_pn + b1 * r_np
        if len(self.layer_next) > 1:
            r_neg = [keras_layers.Add()([times_beta0(pn), times_beta1(np)]) for pn, np in zip(r_pn, r_np)]
            ret = [keras_layers.Subtract()([a, b]) for a, b in zip(r_pos, r_neg)]
            ret = keras_layers.Add()(ret)
        else:
            r_neg = keras_layers.Add()([times_beta0(r_pn), times_beta1(r_np)])
            ret = keras_layers.Subtract()([r_pos, r_neg])
        return ret

class AlphaBetaX1000Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1000Rule, self).__init__(*args,
                                                 alpha=(1, 0),
                                                 beta=(0, 0),
                                                 bias=True,
                                                 **kwargs)

class AlphaBetaX1010Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1010Rule, self).__init__(*args,
                                                 alpha=(1, 0),
                                                 beta=(0, -1),
                                                 bias=True,
                                                 **kwargs)

class AlphaBetaX1001Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1001Rule, self).__init__(*args,
                                                 alpha=(1, 1),
                                                 beta=(0, 0),
                                                 bias=True,
                                                 **kwargs)

class AlphaBetaX2m100Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX2m100Rule, self).__init__(*args,
                                                  alpha=(2, 0),
                                                  beta=(1, 0),
                                                  bias=True,
                                                  **kwargs)

class BoundedRule(reverse_map.ReplacementLayer):
    """Z_B rule from the Deep Taylor Decomposition"""
    # TODO: this only works for relu networks, needs to be extended.
    # TODO: check

    def __init__(self,
                 layer,
                 *args,
                 low=-1,
                 high=1,
                 copy_weights=False,
                 **kwargs):
        self._low = low
        self._high = high

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
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

        super(BoundedRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        to_low = keras_layers.Lambda(lambda x: x * 0 + self._low)
        to_high = keras_layers.Lambda(lambda x: x * 0 + self._high)
        low = [to_low(x) for x in ins]
        high = [to_high(x) for x in ins]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            tape.watch(low)
            tape.watch(high)
            outs = self.layer_func(ins)
            # Get values for the division.
            A = self._layer_wo_act(ins)
            B = self._layer_wo_act_positive(low)
            C = self._layer_wo_act_negative(high)
            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                A = self._neuron_sel_and_head_map(A, neuron_selection, r_init)
                B = self._neuron_sel_and_head_map(B, neuron_selection, r_init)
                C = self._neuron_sel_and_head_map(C, neuron_selection, r_init)

        return outs, low, high, A, B, C, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, low, high, A, B, C, tape = args

        if reversed_outs is None:
            reversed_outs = outs

        if len(self.layer_next) > 1:
            Zs = [keras_layers.Subtract()([a, keras_layers.Add()([b, c])]) for a, b, c in zip(A, B, C)]
            # Divide relevances with the value.
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
            # Distribute along the gradient.
            tmpA = [tape.gradient(A, ins, output_gradients=t) for t in tmp]
            tmpB = [tape.gradient(B, ins, output_gradients=t) for t in tmp]
            tmpC = [tape.gradient(C, ins, output_gradients=t) for t in tmp]
            ret = keras_layers.Add()([keras_layers.Subtract()([a, keras_layers.Add()([b, c])]) for a, b, c in zip(tmpA, tmpB, tmpC)])
        else:
            Zs = keras_layers.Subtract()([A, keras_layers.Add()([B, C])])
            # Divide relevances with the value.
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])
            # Distribute along the gradient.
            tmpA = tape.gradient(A, ins, output_gradients=tmp)
            tmpB = tape.gradient(B, low, output_gradients=tmp)
            tmpC = tape.gradient(C, high, output_gradients=tmp)
            ret = keras_layers.Subtract()([tmpA, keras_layers.Add()([tmpB, tmpC])])

        return ret


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

class ZPlusFastRule(reverse_map.ReplacementLayer):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """

    def __init__(self, layer, *args, copy_weights=False, **kwargs):
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

        super(ZPlusFastRule, self).__init__(layer, *args, **kwargs)

    def apply(self, ins, neuron_selection):
        pass

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            # Get activations.
            Zs = self._layer_wo_act_b_positive(ins)
            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)

        return outs, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, Zs, tape = args

        if reversed_outs is None:
            reversed_outs = outs

        if len(self.layer_next) > 1:
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp_1 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
            # Re-weight relevance with the input values.
            ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp_1])
        else:
            # Divide incoming relevance by the activations.
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp_1 = tape.gradient(Zs, ins, output_gradients=tmp)
            # Re-weight relevance with the input values.
            ret = keras_layers.Multiply()([ins, tmp_1])

        return ret

class GammaRule(reverse_map.ReplacementLayer):
    """
    TODO add documentation
    """


    def __init__(self,
                 layer,
                 *args,
                 gamma=None,
                 bias=True,
                 copy_weights=False,
                 **kwargs):

        if gamma is None:
            raise ValueError("Invalid gamma for LRP-Gamma Rule: " + str(gamma) + ". Please provide a valid gamma value")

        self._gamma = gamma

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
        if copy_weights:
            weights = layer.get_weights()
            if not bias and layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
        else:
            weights = layer.weights
            if not bias and layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * iK.to_floatx(x > 0) for x in weights]

        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=weights,
            name_template="reversed_kernel_%s")

        super(GammaRule, self).__init__(layer, *args, **kwargs)

    def apply(self, ins, neuron_selection):
        pass

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
        ins_pos = keep_positives(ins)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            tape.watch(ins_pos)
            outs = self.layer_func(ins)
            Zs_pos = self._layer_wo_act_positive(ins_pos)
            Zs_act = self._layer_wo_act(ins)
            Zs_pos_act = self._layer_wo_act(ins_pos)
            Zs_act_pos = self._layer_wo_act_positive(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs_pos = self._neuron_sel_and_head_map(Zs_pos, neuron_selection, r_init)
                Zs_act = self._neuron_sel_and_head_map(Zs_act, neuron_selection, r_init)
                Zs_pos_act = self._neuron_sel_and_head_map(Zs_pos_act, neuron_selection, r_init)
                Zs_act_pos = self._neuron_sel_and_head_map(Zs_act_pos, neuron_selection, r_init)

        return outs, ins_pos, Zs_pos, Zs_act, Zs_pos_act, Zs_act_pos, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs, ins_pos, Zs_pos, Zs_act, Zs_pos_act, Zs_act_pos, tape = args
        # this method is correct, but wasteful
        times_gamma = keras_layers.Lambda(lambda x: x * self._gamma)

        def f(i1, i2, z1, z2, rev):

            if rev is None:
                rev = outs

            Zs = keras_layers.Add()([z1, z2])

            # Divide incoming relevance by the activations.
            if len(self.layer_next) > 1:
                tmp = [ilayers.SafeDivide()([r, Zs]) for r in rev]
                # Propagate the relevance to the input neurons
                # using the gradient
                tmp1 = [tape.gradient(z1, i1, output_gradients=t) for t in tmp]
                tmp2 = [tape.gradient(z2, i2, output_gradients=t) for t in tmp]
                # Re-weight relevance with the input values.
                tmp_1 = [keras_layers.Multiply()([i1, t]) for t in tmp1]
                tmp_2 = [keras_layers.Multiply()([i2, t]) for t in tmp2]
                # combine
                combined = [keras_layers.Add()([a, b]) for a, b in zip(tmp_1, tmp_2)]
            else:
                tmp = ilayers.SafeDivide()([rev, Zs])
                # Propagate the relevance to the input neurons
                # using the gradient
                tmp1 = tape.gradient(z1, i1, output_gradients=tmp)
                tmp2 = tape.gradient(z2, i2, output_gradients=tmp)
                # Re-weight relevance with the input values.

                tmp_1 = keras_layers.Multiply()([i1, tmp1])
                tmp_2 = keras_layers.Multiply()([i2, tmp2])
                # combine
                combined = keras_layers.Add()([tmp_1, tmp_2])
            return combined

        # xpos*wpos + xact*wact
        activator_relevances = f(ins_pos, ins, Zs_pos, Zs_act, reversed_outs)
        # xpos*wact + xact*wpos
        all_relevances = f(ins_pos, ins, Zs_pos_act, Zs_act_pos, reversed_outs)

        if len(self.layer_next) > 1:
            sub = [keras_layers.Subtract()([times_gamma(a), b]) for a, b in zip(activator_relevances, all_relevances)]
            ret = keras_layers.Add()(sub)
        else:
            ret = keras_layers.Subtract()([times_gamma(activator_relevances), all_relevances])
        return ret

# TODO not tested in tf2.0 yet
class BatchNormalizationReverseRule(reverse_map.ReplacementLayer):
    """Special BN handler that applies the Z-Rule"""

    def __init__(self, layer, *args, **kwargs):
        config = layer.get_config()

        self._center = config['center']
        self._scale = config['scale']
        self._axis = config['axis']

        self._mean = layer.moving_mean
        self._std = layer.moving_variance
        if self._center:
            self._beta = layer.beta
        super(BatchNormalizationReverseRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        outs = self.layer_func(ins)

        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)

        return outs

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        outs = args

        if reversed_outs is None:
            reversed_outs = outs

        # prepare broadcasting shape for layer parameters
        broadcast_shape = [1] * len(self.input_shape[0])
        broadcast_shape[self._axis] = self.input_shape[0][self._axis]
        broadcast_shape[0] = -1

        # reweight relevances as
        #        x * (y - beta)     R
        # Rin = ---------------- * ----
        #           x - mu          y
        # batch norm can be considered as 3 distinct layers of subtraction,
        # multiplication and then addition. The multiplicative scaling layer
        # has no effect on LRP and functions as a linear activation layer

        minus_mu = keras_layers.Lambda(lambda x: x - K.reshape(self._mean, broadcast_shape))
        minus_beta = keras_layers.Lambda(lambda x: x - K.reshape(self._beta, broadcast_shape))
        prepare_div = keras_layers.Lambda(
            lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * K.epsilon())

        x_minus_mu = minus_mu(ins)
        if self._center:
            y_minus_beta = [minus_beta(o) for o in outs]
        else:
            y_minus_beta = outs

        if len(self.layer_next) > 1:

            numerator = [keras_layers.Multiply()([ins, y_minus_beta, r]) for r in reversed_outs]
            denominator = keras_layers.Multiply()([x_minus_mu, outs])

            ret = keras_layers.Add()([ilayers.SafeDivide()([n, prepare_div(denominator)]) for n in numerator])
        else:

            numerator = keras_layers.Multiply()([ins, y_minus_beta, reversed_outs])
            denominator = keras_layers.Multiply()([x_minus_mu, outs])
            ret = ilayers.SafeDivide()([numerator, prepare_div(denominator)])

        return ret

# TODO not tested in tf2.0 yet
class AddReverseRule(reverse_map.ReplacementLayer):
    """Special Add layer handler that applies the Z-Rule"""

    def __init__(self, layer, *args, **kwargs):
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             name_template="no_act_%s")
        super(AddReverseRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Zs = self._layer_wo_act(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)

        return outs, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):

        # the outputs of the pooling operation at each location is the sum of its inputs.
        # the forward message must be known in this case, and are the inputs for each pooling thing.
        # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
        # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt

        outs, Zs, tape = args
        # last layer
        if reversed_outs is None:
            reversed_outs = Zs

        # Divide incoming relevance by the activations.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
            # Propagate the relevance to input neurons
            # using the gradient.
            if len(self.input_shape) > 1:
                tmp2 = [[tape.gradient(Zs, i, output_gradients=t) for t in tmp] for i in ins]
                ret = [keras_layers.Add()([keras_layers.Multiply()([i, t]) for t in tmp2[idx]]) for idx, i in enumerate(ins)]
            else:
                tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
                ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])
            # Propagate the relevance to input neurons
            # using the gradient.
            if len(self.input_shape) > 1:
                tmp2 = [tape.gradient(Zs, i, output_gradients=tmp) for i in ins]
                ret = [keras_layers.Multiply()([i, tmp2[idx]]) for idx, i in enumerate(ins)]
            else:
                tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
                ret = keras_layers.Multiply()([ins, tmp2])

        return ret

# TODO not tested in tf2.0 yet
class AveragePoolingReverseRule(reverse_map.ReplacementLayer):
    """Special AveragePooling handler that applies the Z-Rule"""

    def __init__(self, layer, *args, **kwargs):
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             name_template="no_act_%s")
        super(AveragePoolingReverseRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Zs = self._layer_wo_act(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)

        return outs, Zs, tape

    def explain_hook(self, ins, reversed_outs, args):

        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        # the outputs of the pooling operation at each location is the sum of its inputs.
        # the forward message must be known in this case, and are the inputs for each pooling thing.
        # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
        # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt

        uts, Zs, tape = args
        # last layer
        if reversed_outs is None:
            reversed_outs = Zs

        # Divide incoming relevance by the activations.
        if len(self.layer_next) > 1:
            tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
            ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
        else:
            tmp = ilayers.SafeDivide()([reversed_outs, Zs])
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
            ret = keras_layers.Multiply()([ins, tmp2])

        return ret
