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
from ...utils.keras import functional as kfunctional

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

layer_mapping = {}

#---------------------------------------------------Rule Classes------------------------------------

class ZRule(reverse_map.ReplacementLayer):
    def __init__(self, layer, *args, **kwargs):
        bias = kwargs.pop("bias", True)

        #this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, bias) in layer_mapping.keys():
            self._layer_wo_act = layer_mapping[(layer.name, type(self).__name__, bias)]
        else:
            self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             keep_bias=bias,
                                                             name_template="no_act_%s")
            layer_mapping[(layer.name, type(self).__name__, bias)] = self._layer_wo_act

        self._explain_func = None
        #print(self._layer_wo_act.get_config()["use_bias"])
        super(ZRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_zrule_explanation
        else:
            self._explain_func = kfunctional.zrule_explanation

    def compute_explanation(self, ins, reversed_outs):

        #some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals["stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     )
        else:
            ret = self._explain_func(ins, self._layer_wo_act, self._out_func, reversed_outs, len(self.input_shape),
                                     len(self.layer_next))

        # apply correct explanation function
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

        # this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, bias) in layer_mapping.keys():
            self._layer_wo_act = layer_mapping[(layer.name, type(self).__name__, bias)]
        else:
            self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                                 keep_bias=bias,
                                                                 name_template="no_act_%s")
            layer_mapping[(layer.name, type(self).__name__, bias)] = self._layer_wo_act
        super(EpsilonRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_epsilonrule_explanation
        else:
            self._explain_func = kfunctional.epsilonrule_explanation

    def compute_explanation(self, ins, reversed_outs):

        # some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (
                self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals[
            "stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     self._epsilon
                                     )
        else:
            ret = self._explain_func(ins, self._layer_wo_act, self._out_func, reversed_outs, len(self.input_shape),
                                     len(self.layer_next), self._epsilon)

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

        # this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, False) in layer_mapping.keys():
            self._layer_wo_act_b = layer_mapping[(layer.name, type(self).__name__, False)]
        else:
            self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
                layer,
                keep_bias=False,
                weights=weights,
                name_template="reversed_kernel_%s")
            layer_mapping[(layer.name, type(self).__name__, False)] = self._layer_wo_act_b
        super(WSquareRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_wsquarerule_explanation
        else:
            self._explain_func = kfunctional.wsquarerule_explanation

    def compute_explanation(self, ins, reversed_outs):

        # some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (
                self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals[
            "stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act_b,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     )
        else:
            ret = self._explain_func(ins, self._layer_wo_act_b, self._out_func, reversed_outs, len(self.input_shape),
                                     len(self.layer_next))

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

        # this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, False) in layer_mapping.keys():
            self._layer_wo_act_b = layer_mapping[(layer.name, type(self).__name__, False)]
        else:
            #print((layer.name, type(self).__name__, False))
            self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
                layer,
                keep_bias=False,
                weights=weights,
                name_template="reversed_kernel_%s")
            layer_mapping[(layer.name, type(self).__name__, False)] = self._layer_wo_act_b

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

        # this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, bias) in layer_mapping.keys():
            self._layer_wo_act_positive = layer_mapping[(layer.name, type(self).__name__, bias)][0]
            self._layer_wo_act_negative = layer_mapping[(layer.name, type(self).__name__, bias)][1]
        else:
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
            layer_mapping[(layer.name, type(self).__name__, bias)] = [self._layer_wo_act_positive, self._layer_wo_act_negative]

        super(AlphaBetaRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_alphabetarule_explanation
        else:
            self._explain_func = kfunctional.alphabetarule_explanation

    def compute_explanation(self, ins, reversed_outs):

        # some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (
                self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals[
            "stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act_positive,
                                     self._layer_wo_act_negative,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     self._alpha,
                                     self._beta
                                     )
        else:
            ret = self._explain_func(ins,
                                     self._layer_wo_act_positive,
                                     self._layer_wo_act_negative,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self._alpha,
                                     self._beta
                                     )

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

            # this avoids creating a new object each time and reduces tracing
            if (layer.name, type(self).__name__, bias) in layer_mapping.keys():
                self._layer_wo_act_positive = layer_mapping[(layer.name, type(self).__name__, bias)][0]
                self._layer_wo_act_negative = layer_mapping[(layer.name, type(self).__name__, bias)][1]
            else:
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
                layer_mapping[(layer.name, type(self).__name__, bias)] = [self._layer_wo_act_positive, self._layer_wo_act_negative]

        super(AlphaBetaXRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_alphabetaxrule_explanation
        else:
            self._explain_func = kfunctional.alphabetaxrule_explanation

    def compute_explanation(self, ins, reversed_outs):

        # some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (
                self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals[
            "stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act_positive,
                                     self._layer_wo_act_negative,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     self._alpha,
                                     self._beta
                                     )
        else:
            ret = self._explain_func(ins,
                                     self._layer_wo_act_positive,
                                     self._layer_wo_act_negative,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self._alpha,
                                     self._beta
                                     )

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

        # this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, False) in layer_mapping.keys():
            self._layer_wo_act = layer_mapping[(layer.name, type(self).__name__, False)][0]
            self._layer_wo_act_positive = layer_mapping[(layer.name, type(self).__name__, False)][1]
            self._layer_wo_act_negative = layer_mapping[(layer.name, type(self).__name__, False)][2]
        else:
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
            layer_mapping[(layer.name, type(self).__name__, False)] = [self._layer_wo_act, self._layer_wo_act_positive, self._layer_wo_act_negative]

        super(BoundedRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_boundedrule_explanation
        else:
            self._explain_func = kfunctional.boundedrule_explanation

    def compute_explanation(self, ins, reversed_outs):

        # some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (
                self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals[
            "stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act,
                                     self._layer_wo_act_positive,
                                     self._layer_wo_act_negative,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     self._low,
                                     self._high
                                     )
        else:
            ret = self._explain_func(ins,
                                     self._layer_wo_act,
                                     self._layer_wo_act_positive,
                                     self._layer_wo_act_negative,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self._low,
                                     self._high
                                     )

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

        # this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, False) in layer_mapping.keys():
            self._layer_wo_act_b_positive = layer_mapping[(layer.name, type(self).__name__, False)]
        else:
            self._layer_wo_act_b_positive = kgraph.copy_layer_wo_activation(
                layer,
                keep_bias=False,
                weights=weights,
                name_template="reversed_kernel_positive_%s")
            layer_mapping[(layer.name, type(self).__name__, False)] = self._layer_wo_act_b_positive

        super(ZPlusFastRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_zplusfastrule_explanation
        else:
            self._explain_func = kfunctional.zplusfastrule_explanation

    def compute_explanation(self, ins, reversed_outs):

        # some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (
                self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals[
            "stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act_b_positive,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"]
                                     )
        else:
            ret = self._explain_func(ins,
                                     self._layer_wo_act_b_positive,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next)
                                     )

        return ret


class GammaRule(reverse_map.ReplacementLayer):
    """
    The Gamma-Rule
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

        # this avoids creating a new object each time and reduces tracing
        if (layer.name, type(self).__name__, bias) in layer_mapping.keys():
            self._layer_wo_act_positive = layer_mapping[(layer.name, type(self).__name__, bias)][0]
            self._layer_wo_act = layer_mapping[(layer.name, type(self).__name__, bias)][1]
        else:
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
            layer_mapping[(layer.name, type(self).__name__, bias)] = [self._layer_wo_act_positive, self._layer_wo_act]

        super(GammaRule, self).__init__(layer, *args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (
                stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_gammarule_explanation
        else:
            self._explain_func = kfunctional.gammarule_explanation

    def compute_explanation(self, ins, reversed_outs):

        # some preparation
        if len(self.input_shape) > 1:
            raise ValueError("This Layer should only have one input!")

        if reversed_outs is None:
            reversed_outs = self.saved_forward_vals["outs"]

        if len(self.layer_next) == 0 or (
                self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals[
            "stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self._layer_wo_act,
                                     self._layer_wo_act_positive,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     self._gamma
                                     )
        else:
            ret = self._explain_func(ins,
                                     self._layer_wo_act,
                                     self._layer_wo_act_positive,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self._gamma
                                     )

        return ret


#---------------------------------------------------------------
# below cases will be handled by canonizer
# not updating below code for now.
# TODO: delete completely
#---------------------------------------------------------------

#
# class BatchNormalizationReverseRule(reverse_map.ReplacementLayer):
#     """Special BN handler that applies the Z-Rule"""
#
#     def __init__(self, layer, *args, **kwargs):
#         config = layer.get_config()
#
#         self._center = config['center']
#         self._scale = config['scale']
#         self._axis = config['axis']
#
#         self._mean = layer.moving_mean
#         self._std = layer.moving_variance
#         if self._center:
#             self._beta = layer.beta
#         super(BatchNormalizationReverseRule, self).__init__(layer, *args, **kwargs)
#
#     def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
#         outs = self.layer_func(ins)
#
#         # check if final layer (i.e., no next layers)
#         if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
#             outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
#
#         return outs
#
#     def compute_explanation(self, ins, reversed_outs, args):
#
#         if len(self.input_shape) > 1:
#             raise ValueError("This Layer should only have one input!")
#
#         outs = args
#
#         if reversed_outs is None:
#             reversed_outs = outs
#
#         # prepare broadcasting shape for layer parameters
#         broadcast_shape = [1] * len(self.input_shape[0])
#         broadcast_shape[self._axis] = self.input_shape[0][self._axis]
#         broadcast_shape[0] = -1
#
#         # reweight relevances as
#         #        x * (y - beta)     R
#         # Rin = ---------------- * ----
#         #           x - mu          y
#         # batch norm can be considered as 3 distinct layers of subtraction,
#         # multiplication and then addition. The multiplicative scaling layer
#         # has no effect on LRP and functions as a linear activation layer
#
#         minus_mu = keras_layers.Lambda(lambda x: x - K.reshape(self._mean, broadcast_shape))
#         minus_beta = keras_layers.Lambda(lambda x: x - K.reshape(self._beta, broadcast_shape))
#         prepare_div = keras_layers.Lambda(
#             lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * K.epsilon())
#
#         x_minus_mu = minus_mu(ins)
#         if self._center:
#             y_minus_beta = [minus_beta(o) for o in outs]
#         else:
#             y_minus_beta = outs
#
#         if len(self.layer_next) > 1:
#
#             numerator = [keras_layers.Multiply()([ins, y_minus_beta, r]) for r in reversed_outs]
#             denominator = keras_layers.Multiply()([x_minus_mu, outs])
#
#             ret = keras_layers.Add()([ilayers.SafeDivide()([n, prepare_div(denominator)]) for n in numerator])
#         else:
#
#             numerator = keras_layers.Multiply()([ins, y_minus_beta, reversed_outs])
#             denominator = keras_layers.Multiply()([x_minus_mu, outs])
#             ret = ilayers.SafeDivide()([numerator, prepare_div(denominator)])
#
#         return ret
#
# class AddReverseRule(reverse_map.ReplacementLayer):
#     """Special Add layer handler that applies the Z-Rule"""
#
#     def __init__(self, layer, *args, **kwargs):
#         self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
#                                                              name_template="no_act_%s")
#         super(AddReverseRule, self).__init__(layer, *args, **kwargs)
#
#     def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
#         with tf.GradientTape(persistent=True) as tape:
#             tape.watch(ins)
#             outs = self.layer_func(ins)
#             Zs = self._layer_wo_act(ins)
#
#             # check if final layer (i.e., no next layers)
#             if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
#                 outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
#                 Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)
#
#         return outs, Zs, tape
#
#     def compute_explanation(self, ins, reversed_outs, args):
#
#         # the outputs of the pooling operation at each location is the sum of its inputs.
#         # the forward message must be known in this case, and are the inputs for each pooling thing.
#         # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
#         # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt
#
#         outs, Zs, tape = args
#         # last layer
#         if reversed_outs is None:
#             reversed_outs = Zs
#
#         # Divide incoming relevance by the activations.
#         if len(self.layer_next) > 1:
#             tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
#             # Propagate the relevance to input neurons
#             # using the gradient.
#             if len(self.input_shape) > 1:
#                 tmp2 = [[tape.gradient(Zs, i, output_gradients=t) for t in tmp] for i in ins]
#                 ret = [keras_layers.Add()([keras_layers.Multiply()([i, t]) for t in tmp2[idx]]) for idx, i in enumerate(ins)]
#             else:
#                 tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
#                 ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
#         else:
#             tmp = ilayers.SafeDivide()([reversed_outs, Zs])
#             # Propagate the relevance to input neurons
#             # using the gradient.
#             if len(self.input_shape) > 1:
#                 tmp2 = [tape.gradient(Zs, i, output_gradients=tmp) for i in ins]
#                 ret = [keras_layers.Multiply()([i, tmp2[idx]]) for idx, i in enumerate(ins)]
#             else:
#                 tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
#                 ret = keras_layers.Multiply()([ins, tmp2])
#
#         return ret
#
# class AveragePoolingReverseRule(reverse_map.ReplacementLayer):
#     """Special AveragePooling handler that applies the Z-Rule"""
#
#     def __init__(self, layer, *args, **kwargs):
#         self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
#                                                              name_template="no_act_%s")
#         super(AveragePoolingReverseRule, self).__init__(layer, *args, **kwargs)
#
#     def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
#         with tf.GradientTape(persistent=True) as tape:
#             tape.watch(ins)
#             outs = self.layer_func(ins)
#             Zs = self._layer_wo_act(ins)
#
#             # check if final layer (i.e., no next layers)
#             if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
#                 outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
#                 Zs = self._neuron_sel_and_head_map(Zs, neuron_selection, r_init)
#
#         return outs, Zs, tape
#
#     def compute_explanation(self, ins, reversed_outs, args):
#
#         if len(self.input_shape) > 1:
#             raise ValueError("This Layer should only have one input!")
#
#         # the outputs of the pooling operation at each location is the sum of its inputs.
#         # the forward message must be known in this case, and are the inputs for each pooling thing.
#         # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
#         # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt
#
#         uts, Zs, tape = args
#         # last layer
#         if reversed_outs is None:
#             reversed_outs = Zs
#
#         # Divide incoming relevance by the activations.
#         if len(self.layer_next) > 1:
#             tmp = [ilayers.SafeDivide()([r, Zs]) for r in reversed_outs]
#             # Propagate the relevance to input neurons
#             # using the gradient.
#             tmp2 = [tape.gradient(Zs, ins, output_gradients=t) for t in tmp]
#             ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
#         else:
#             tmp = ilayers.SafeDivide()([reversed_outs, Zs])
#             # Propagate the relevance to input neurons
#             # using the gradient.
#             tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
#             ret = keras_layers.Multiply()([ins, tmp2])
#
#         return ret
