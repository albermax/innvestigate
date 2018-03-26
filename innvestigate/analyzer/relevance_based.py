# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import inspect
import keras
import keras.backend as K
import keras.engine.topology
import keras.models
import keras.layers
import keras.layers.convolutional
import keras.layers.core
import keras.layers.local
import keras.layers.noise
import keras.layers.normalization
import keras.layers.pooling
import numpy as np


from . import base
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph


__all__ = [
    "BaselineLRPZ",

    "LRP_RULES",

    "LRP",

    "LRPZ",
    "LRPZIgnoreBias",

    "LRPEpsilon",
    "LRPEpsilonIgnoreBias",

    "LRPWSquare",
    "LRPFlat",

    "LRPAlphaBeta",

    "LRPAlpha2Beta1",
    "LRPAlpha2Beta1IgnoreBias",
    "LRPAlpha1Beta0",
    "LRPAlpha1Beta0IgnoreBias",
    "LRPZPlus",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineLRPZ(base.AnalyzerNetworkBase):
    # TODO: Inherit from LRP, specialize from there.
    """LRPZ analyzer.

    Applies the "LRP-Z" algorithm to analyze the model.
    Based on the gradient times the input formula.
    This formula holds only for ReLU/MaxPooling networks, for which
    LRP-Z collapses into the stated formula.

    :param model: A Keras model.
    :param allow_lambda_layers: Approximate lambda layers with the
      gradient.
    """

    def __init__(self, model, allow_lambda_layers=False, **kwargs):
        # Inside function to not break import if Keras changes.
        BASELINELRPZ_LAYERS = (
            keras.engine.topology.InputLayer,
            keras.layers.convolutional.Conv1D,
            keras.layers.convolutional.Conv2D,
            keras.layers.convolutional.Conv2DTranspose,
            keras.layers.convolutional.Conv3D,
            keras.layers.convolutional.Conv3DTranspose,
            keras.layers.convolutional.Cropping1D,
            keras.layers.convolutional.Cropping2D,
            keras.layers.convolutional.Cropping3D,
            keras.layers.convolutional.SeparableConv1D,
            keras.layers.convolutional.SeparableConv2D,
            keras.layers.convolutional.UpSampling1D,
            keras.layers.convolutional.UpSampling2D,
            keras.layers.convolutional.UpSampling3D,
            keras.layers.convolutional.ZeroPadding1D,
            keras.layers.convolutional.ZeroPadding2D,
            keras.layers.convolutional.ZeroPadding3D,
            keras.layers.core.Activation,
            keras.layers.core.ActivityRegularization,
            keras.layers.core.Dense,
            keras.layers.core.Dropout,
            keras.layers.core.Flatten,
            keras.layers.core.Lambda,
            keras.layers.core.Masking,
            keras.layers.core.Permute,
            keras.layers.core.RepeatVector,
            keras.layers.core.Reshape,
            keras.layers.core.SpatialDropout1D,
            keras.layers.core.SpatialDropout2D,
            keras.layers.core.SpatialDropout3D,
            keras.layers.local.LocallyConnected1D,
            keras.layers.local.LocallyConnected2D,
            keras.layers.Add,
            keras.layers.Concatenate,
            keras.layers.Dot,
            keras.layers.Maximum,
            keras.layers.Minimum,
            keras.layers.Subtract,
            keras.layers.noise.AlphaDropout,
            keras.layers.noise.GaussianDropout,
            keras.layers.noise.GaussianNoise,
            keras.layers.normalization.BatchNormalization,
            keras.layers.pooling.GlobalMaxPooling1D,
            keras.layers.pooling.GlobalMaxPooling2D,
            keras.layers.pooling.GlobalMaxPooling3D,
            keras.layers.pooling.MaxPooling1D,
            keras.layers.pooling.MaxPooling2D,
            keras.layers.pooling.MaxPooling3D,
        )

        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            {
                "check":
                lambda layer: not kchecks.only_relu_activation(layer),
                "type": "exception",
                "message": ("BaselineLRPZ is not working for "
                            "networks with non-ReLU activations."),
            },
            {
                "check":
                lambda layer: not isinstance(layer, BASELINELRPZ_LAYERS),
                "type": "exception",
                "message": ("BaselineLRPZ is only defined for "
                            "certain layers."),
            },
            {
                "check":
                lambda layer: (not allow_lambda_layers and
                               isinstance(layer, keras.layers.core.Lambda)),
                "type": "exception",
                "message": ("Lamda layers are not allowed. "
                            "To allow use allow_lambda_layers kw."),
            },
        ]

        self._allow_lambda_layers = allow_lambda_layers

        super(BaselineLRPZ, self).__init__(model, **kwargs)

    def _create_analysis(self, model):
        gradients = iutils.to_list(ilayers.Gradient()(
            model.inputs+[model.outputs[0], ]))
        return [keras.layers.Multiply()([i, g])
                for i, g in zip(model.inputs, gradients)]

    def _get_state(self):
        state = super(BaselineLRPZ, self)._get_state()
        state.update({"allow_lambda_layers": self._allow_lambda_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        allow_lambda_layers = state.pop("allow_lambda_layers")
        kwargs = super(BaselineLRPZ, clazz)._state_to_kwargs(state)
        kwargs.update({"allow_lambda_layers": allow_lambda_layers})
        return kwargs


###############################################################################
###############################################################################
###############################################################################

def _assert_epsilon_parameter(epsilon, caller):
    """
        Function for asserting epsilon parameter choice
        passed to constructors inheriting from EpsilonRule
        and LRPEpsilon.
        The following conditions can not be met:

        epsilon > 1

        :param epsilon: the epsilon parameter.
        :param caller: the class instance calling this assertion function
    """

    err_head = "Constructor call to {} : ".format(caller.__class__.__name__)
    err_msg = err_head + "Parameter epsilon must be > 0"
    assert epsilon > 0, err_msg
    return epsilon


def _assert_infer_alpha_beta_parameters(alpha, beta, caller):
    """
        Function for asserting parameter choices for alpha and beta
        passed to constructors inheriting from AlphaBetaRule
        and LRPAlphaBeta.

        since alpha - beta are subjected to sum to 1,
        it is sufficient for only one of the parameters to be passed
        to a corresponding class constructor.
        this method will cause an assertion error if both are None
        or the following conditions can not be met

        alpha >= 1
        beta >= 0
        alpha - beta = 1

        :param alpha: the alpha parameter.
        :param beta: the beta parameter
        :param caller: the class instance calling this assertion function
    """

    err_head = "Constructor call to {} : ".format(caller.__class__.__name__)
    err_msg = err_head + "Neither alpha or beta were given"
    assert not (alpha is None and beta is None), err_msg

    #assert passed parameter choices
    if alpha is not None:
        err_msg = err_head +"Passed parameter alpha invalid. Expecting alpha >= 1 but was {}".format(alpha)
        assert alpha >= 1, err_msg

    if beta is not None:
        err_msg = err_head +"Passed parameter beta invalid. Expecting beta >= 0 but was {}".format(beta)
        assert beta >= 0, err_msg

    #assert inferred parameter choices
    if alpha is None:
        alpha = beta + 1
        err_msg = err_head +"Inferring alpha from given beta {} s.t. alpha - beta = 1, with condition alpha >= 1 not possible.".format(beta)
        assert alpha >= 1, err_msg

    if beta is None:
        beta = alpha - 1
        err_msg = err_head +"Inferring beta from given alpha {} s.t. alpha - beta = 1, with condition beta >= 0 not possible.".format(alpha)
        assert beta >= 0, err_msg

    #final check: alpha - beta = 1
    amb = alpha - beta
    err_msg = err_head +"Condition alpha - beta = 1 not fulfilled. alpha={} ; beta={} -> alpha - beta = {}".format(alpha, beta, amb)
    assert amb == 1, err_msg

    #return benign values for alpha and beta
    return alpha, beta

###############################################################################
###############################################################################
###############################################################################



class ZRule(kgraph.ReverseMappingBase):
    """
    Basic LRP decomposition rule (for layers with weight kernels),
    which considers the bias a constant input neuron.
    """

    def __init__(self, layer, state, bias=True):
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             keep_bias=bias,
                                                             name_template="reversed_kernel_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        # Get activations.
        Zs = kutils.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class ZIgnoreBiasRule(ZRule):
    """
    Basic LRP decomposition rule, ignoring the bias neuron
    """
    def __init__(self, *args, **kwargs):
        return super(ZIgnoreBiasRule, self).__init__(*args,
                                                   bias=False,
                                                   **kwargs)


#TODO: make subclass of ZRule
#TODO: fix computation of z+ to not depend on positive inputs, but positive preactivations
class ZPlusRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        # The z-plus rule only works with positive weights and
        # no biases.
        self._layer_wo_act_b_positive = kgraph.copy_layer_wo_activation(
            layer, keep_bias=False,
            name_template="reversed_kernel_positive_%s")
        tmp = [x * (x > 0)
               for x in self._layer_wo_act_b_positive.get_weights()]
        self._layer_wo_act_b_positive.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        # Get activations.
        Zs = kutils.apply(self._layer_wo_act_b_positive, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class EpsilonRule(kgraph.ReverseMappingBase):
    """
    Similar to ZRule.
    The only difference is the addition of a numerical stabilizer term
    epsilon to the decomposition function's denominator.
    the sign of epsilon depends on the sign of the output activation
    0 is considered to be positive, ie sign(0) = 1
    """

    def __init__(self, layer, state, epsilon = 1e-7, bias=True):
        self._epsilon = _assert_epsilon_parameter(epsilon, self)
        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s")


    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # The epsilon rule aligns epsilon with the (extended) sign: 0 is considered to be positive
        prepare_div = keras.layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x,0), K.floatx())*2-1)*self._epsilon)

        # Get activations.
        Zs = kutils.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.Divide()([a, prepare_div(b)])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]



class EpsilonIgnoreBiasRule(EpsilonRule):
    def __init__(self, *args, **kwargs):
        return super(EpsilonIgnoreBiasRule, self).__init__(*args,
                                                         bias=False, **kwargs)


class WSquareRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        # W-square rule works with squared weights and no biases.
        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s")
        tmp = [x**2 for x in self._layer_wo_act_b.get_weights()]
        self._layer_wo_act_b.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # Create dummy forward path to take the derivative below.
        Ys = kutils.apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the squared weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = iutils.to_list(self._layer_wo_act_b(ones))
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.to_list(grad(Xs+Ys+Rs))
        return tmp


# TODO: Make sublcass of WSquare rule
class FlatRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        # The flat rule works with weights equal to one and
        # no biases.
        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s")
        tmp = [np.ones_like(x) for x in self._layer_wo_act_b.get_weights()]
        self._layer_wo_act_b.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # Create dummy forward path to take the derivative below.
        Ys = kutils.apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the one-weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = iutils.to_list(self._layer_wo_act_b(ones))
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.to_list(grad(Xs+Ys+tmp))
        return tmp










# TODO: could potentially inherit from and use ZRule.
class AlphaBetaRule(kgraph.ReverseMappingBase):
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
    #TODO assert alpha beta conditions
    #TODO extend: either give alpha, or beta, or both. if one is given, infer the others.

    # TODO: this only works for relu networks, needs to be extended.
    def __init__(self, layer, state, alpha=None, beta=None, bias=True):
        alpha, beta = _assert_infer_alpha_beta_parameters(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta

        # Positive part works with layer with only positive
        # weights and biases.
        # The negative part does accordingly.
        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            name_template="reversed_kernel_negative_%s")

        #TODO: make decomposition not rely on positive or negative weights, but positive and negative preactivations.
        positive_weights = [x * (x > 0)
                            for x in self._layer_wo_act_positive.get_weights()]
        negative_weights = [x * (x < 0)
                            for x in self._layer_wo_act_negative.get_weights()]
        self._layer_wo_act_positive.set_weights(positive_weights)
        self._layer_wo_act_negative.set_weights(negative_weights)


    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        times_alpha = keras.layers.Lambda(lambda x: x * self._alpha)
        times_beta = keras.layers.Lambda(lambda x: x * self._beta)

        def f(layer):
            # Get activations.
            Zs = kutils.apply(layer, Xs)
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([a, b])
                   for a, b in zip(Rs, Zs)]
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp = iutils.to_list(grad(Xs+Zs+tmp))
            # Re-weight relevance with the input values.
            return [keras.layers.Multiply()([a, b])
                    for a, b in zip(Xs, tmp)]

        # Compute positive and negative relevance.'
        positive_part = f(self._layer_wo_act_positive)
        negative_part = f(self._layer_wo_act_negative)
        # Join weighted positive and negative relevance again.
        return [keras.layers.Subtract()([times_alpha(a), times_beta(b)])
                for a, b in zip(positive_part, negative_part)]



class AlphaBetaIgnoreBiasRule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(AlphaBetaIgnoreBiasRule, self).__init__(*args,
                                                           bias=False,
                                                           **kwargs)



class Alpha2Beta1Rule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha2Beta1Rule, self).__init__(*args,
                                                     alpha=2,
                                                     beta=1,
                                                     bias=True,
                                                     **kwargs)


class Alpha2Beta1IgnoreBiasRule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha2Beta1IgnoreBiasRule, self).__init__(*args,
                                                             alpha=2,
                                                             beta=1,
                                                             bias=False,
                                                             **kwargs)


class Alpha1Beta0Rule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha1Beta0Rule, self).__init__(*args,
                                                     alpha=1,
                                                     beta=0,
                                                     bias=True,
                                                     **kwargs)


class Alpha1Beta0IgnoreBiasRule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha1Beta0IgnoreBiasRule, self).__init__(*args,
                                                             alpha=1,
                                                             beta=0,
                                                             bias=False,
                                                             **kwargs)


class BoundedRule(kgraph.ReverseMappingBase):
    # TODO: this only works for relu networks, needs to be extended.
    # TODO: check
    def __init__(self, layer, state, low=-1, high=1):
        self._low = low
        self._high = high

        # This rule works with three variants of the layer, all without biases.
        # One is the original form and two with only the positive or
        # negative weights.
        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_%s")
        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_negative_%s")
        positive_weights = [x * (x > 0)
                            for x in self._layer_wo_act_positive.get_weights()]
        negative_weights = [x * (x < 0)
                            for x in self._layer_wo_act_negative.get_weights()]
        self._layer_wo_act_positive.set_weights(positive_weights)
        self._layer_wo_act_negative.set_weights(negative_weights)

    # todo: check if this is a correct implementation.
    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        to_low = keras.layers.Lambda(lambda x: x * 0 + self._low)
        to_high = keras.layers.Lambda(lambda x: x * 0 + self._high)

        def f(Xs):
            low = [to_low(x) for x in Xs]
            high = [to_high(x) for x in Xs]

            A = kutils.apply(self._layer_wo_act, Xs)
            B = kutils.apply(self._layer_wo_act_positive, low)
            C = kutils.apply(self._layer_wo_act_negative, high)
            return [keras.layers.Subtract()([a, keras.layers.Add()([b, c])])
                    for a, b, c in zip(A, B, C)]

        # Get values for the division.
        Zs = f(Xs)
        # Divide relevances with the value.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
        return tmp


#        alpha-beta all networks
#        bias+- for some other rules
LRP_RULES = {
    "Z": ZRule,
    "ZIgnoreBias": ZIgnoreBiasRule,

    "Epsilon": EpsilonRule,
    "EpsilonIgnoreBias": EpsilonIgnoreBiasRule,

    "WSquare": WSquareRule,
    "Flat": FlatRule,

    "AlphaBeta": AlphaBetaRule,
    "AlphaBetaIgnoreBias": AlphaBetaIgnoreBiasRule,

    "Alpha2Beta1": Alpha2Beta1Rule,
    "Alpha2Beta1IgnoreBias": Alpha2Beta1IgnoreBiasRule,
    "Alpha1Beta0": Alpha1Beta0Rule,
    "Alpha1Beta0IgnoreBias": Alpha1Beta0IgnoreBiasRule,

    "ZPlus": ZPlusRule,
    "Bounded": BoundedRule,
}


###############################################################################
###############################################################################
###############################################################################


class LRP(base.ReverseAnalyzerBase):
    """
    Base class for LRP-based model analyzers


    :param model: A Keras model.

    :param rule: A rule can be a  string or a Rule object, lists thereof or a list of conditions [(Condition, Rule), ... ]
      gradient.

    :param input_layer_rule: either a Rule object, atuple of (low, high) the min/max pixel values of the inputs
    """

    def __init__(self,
                 model, *args,
                 rule=None,
                 input_layer_rule=None,
                 **kwargs):
        self._model_checks = [
            # TODO: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            {
                "check": lambda layer: not kchecks.is_convnet_layer(layer),
                "type": "warning",
                "message": ("LRP is only tested for "
                            "convolutional neural networks."),
            },
        ]


        # check if rule was given explicitly.
        # TODO: make LRP-Z the default behaviour / rule
        # TODO: create PRESETs for LRP, e.g. alphabeta for conv, eps/z für linear layers
        # rule can be a string, a list (of strings) or a list of conditions [(Condition, Rule), ... ] for each layer.
        if rule is None:
            raise ValueError("Need LRP rule(s).")



        if isinstance(rule, list):
            # copy refrences
            self._rule = list(rule)
        else:
            self._rule = rule
        self._input_layer_rule = input_layer_rule


        if(
           isinstance(rule, six.string_types) or
           (inspect.isclass(rule) and issubclass(rule, kgraph.ReverseMappingBase)) # NOTE: All LRP rules inherit from kgraph.ReverseMappingBase
        ):
            # the given rule is a single strig or single rule implementing class
            use_conditions = True
            rules = [(lambda a, b: True, rule)]

        elif not isinstance(rule[0], tuple):
            # rule list of rule strings or classes
            use_conditions = False
            rules = list(rule)
        else:
            # rule is list of conditioned rules
            use_conditions = True
            rules = rule


        #TODO: find out: what are the assumptions here?
        if self._input_layer_rule is not None:
            input_layer_rule = self._input_layer_rule
            if isinstance(input_layer_rule, tuple):
                low, high = input_layer_rule

                # TODO avoid ad-hoc class definition here. Try to use Bounded Rule.
                class input_layer_rule(BoundedRule):
                    def __init__(self, *args, **kwars):
                        return super(input_layer_rule, self).__init__(
                            *args, low=low, high=high, **kwargs)


            if use_conditions is True:
                rules.insert(0,
                             (lambda layer, foo: kgraph.is_input_layer(layer),
                              input_layer_rule))
            else:
                rules.insert(0, input_layer_rule)


        def select_rule(layer, reverse_state):
            # TODO: check if use_conditions functions properly. should be class variable.
            if use_conditions is True:
                for condition, rule in rules:
                    if condition(layer, reverse_state):
                        return rule
                raise Exception("No rule applies to layer: %s" % layer)
            else:
                return rules.pop()


        class ReverseLayer(kgraph.ReverseMappingBase):
            # TODO: refactor as independent class?
            def __init__(self, layer, state):
                rule_class = select_rule(layer, state)
                if isinstance(rule_class, six.string_types):
                    rule_class = LRP_RULES[rule]
                self._rule = rule_class(layer, state) #TODO. add def call() to Rule base class, which is a setter for layer, state, to avoid above ad-hoc-input rule class gen.

            def apply(self, Xs, Ys, Rs, reverse_state):
                #print(reverse_state['layer'].__class__.__name__, 'Rule.apply kicking in for rule', self._rule)
                return self._rule.apply(Xs, Ys, Rs, reverse_state)


        # conditional mappings layer_criterion -> Rule on how to handle backward passes through layers.
        self._conditional_mappings = [
            (kchecks.contains_kernel, ReverseLayer),
        ]

        # finalize constructor.
        return super(LRP, self).__init__(model, *args, **kwargs)




    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        #print(reverse_state['layer'].__class__.__name__, '_default_reverse_layer kicking in', end=':')
        if(len(Xs) == len(Ys) and
           all([K.int_shape(x) == K.int_shape(y) for x, y in zip(Xs, Ys)])):
        #if isinstance(reverse_state['layer'], keras.layers.Activation): # TODO: complete this. Activation should not be everything.
            # TODO: this is not necessarily true. Do explicit layer check.
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            #print(' just return') #TODO:DEBUG
            return reversed_Ys
        else:
            # TODO: make this more clear, here we assume to have reshape layers
            # TODO: add assert
            # TODO: BatchNorm layer should end up here (?): Implements an affine transformation.
            # TODO: Confirm that behaviour of GradientWRT. Flatten and BatchNorm are correct
            #print(' ilayers.GradientWRT') #TODO:DEBUG
            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)


    def _get_state(self):
        state = super(LRP, self)._get_state()
        state.update({"rule": self._rule})
        state.update({"input_layer_rule": self._input_layer_rule})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        rule = state.pop("rule")
        input_layer_rule = state.pop("input_layer_rule")
        kwargs = super(LRP, clazz)._state_to_kwargs(state)
        kwargs.update({"rule": rule,
                       "input_layer_rule": input_layer_rule})
        return kwargs


###############################################################################
# ANALYZER CLASSES ################################################################
###############################################################################


class _LRPFixedParams(LRP):

    @classmethod
    def _state_to_kwargs(clazz, state):
        kwargs = super(_LRPFixedParams, clazz)._state_to_kwargs(state)
        del kwargs["rule"]
        return kwargs


class LRPZ(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPZ, self).__init__(model, *args, rule="Z", **kwargs)


class LRPZIgnoreBias(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPZIgnoreBias, self).__init__(model, *args,
                                                  rule="ZIgnoreBias", **kwargs)


class LRPZPlus(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPZPlus, self).__init__(model, *args,
                                              rule="ZPlus", **kwargs)


class LRPEpsilon(_LRPFixedParams):

    def __init__(self, model, epsilon=1e-7, bias=True, *args, **kwargs):
        epsilon = _assert_epsilon_parameter(epsilon, self)
        self._epsilon = epsilon

        class EpsilonProxyRule(EpsilonRule):
            """
            Dummy class inheriting from EpsilonRule
            for passing along the chosen parameters from
            the LRP analyzer class to the decopmosition rules.
            """
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=epsilon,
                                                       bias=bias,
                                                       **kwargs)

        return super(LRPEpsilon, self).__init__(model, *args,
                                                  rule=EpsilonProxyRule,
                                                  **kwargs)


class LRPEpsilonIgnoreBias(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPEpsilonIgnoreBias, self).__init__(model, *args,
                                                        rule="EpsilonIgnoreBias",
                                                        **kwargs)


class LRPWSquare(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPWSquare, self).__init__(model, *args,
                                                rule="WSquare", **kwargs)


class LRPFlat(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPFlat, self).__init__(model, *args,
                                             rule="Flat", **kwargs)


#TODO: class for assigning LRPAlphaBeta21 to conv layers and eps to dense layers


class LRPAlphaBeta(LRP):
    """ Base class for LRP AlphaBeta"""

    def __init__(self, model, alpha=None, beta=None, bias=True, *args, **kwargs):
        alpha, beta = _assert_infer_alpha_beta_parameters(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta
        self._bias = bias

        class AlphaBetaProxyRule(AlphaBetaRule):
            """
            Dummy class inheriting from AlphaBetaRule
            for the purpose of passing along the chosen parameters from
            the LRP analyzer class to the decopmosition rules.
            """
            def __init__(self, *args, **kwargs):
                super(AlphaBetaProxyRule, self).__init__(*args,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias,
                                                              **kwargs)

        return super(LRPAlphaBeta, self).__init__(model, *args,
                                                  rule=AlphaBetaProxyRule,
                                                  **kwargs)

    def _get_state(self):
        state = super(LRPAlphaBeta, self)._get_state()
        del state["rule"]
        state.update({"alpha": self._alpha})
        state.update({"beta": self._beta})
        state.update({"bias": self._bias})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        alpha = state.pop("alpha")
        beta = state.pop("beta")
        bias = state.pop("bias")
        state["rule"] = None
        kwargs = super(LRPAlphaBeta, clazz)._state_to_kwargs(state)
        del kwargs["rule"]
        kwargs.update({"alpha": alpha,
                       "beta": beta,
                       "bias": bias})
        return kwargs



class _LRPAlphaBetaFixedParams(LRPAlphaBeta):

    @classmethod
    def _state_to_kwargs(clazz, state):
        kwargs = super(_LRPAlphaBetaFixedParams, clazz)._state_to_kwargs(state)
        del kwargs["alpha"]
        del kwargs["beta"]
        del kwargs["bias"]
        return kwargs


###############################################################################
# LRP PARAM PRESETS CLASSES ###################################################
###############################################################################

class LRPAlpha2Beta1(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha2Beta1, self).__init__(model, *args,
                                                    alpha=2,
                                                    beta=1,
                                                    bias=True,
                                                    **kwargs)


class LRPAlpha2Beta1IgnoreBias(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha2Beta1IgnoreBias, self).__init__(model, *args,
                                                            alpha=2,
                                                            beta=1,
                                                            bias=False,
                                                            **kwargs)


class LRPAlpha1Beta0(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta0, self).__init__(model, *args,
                                                    alpha=1,
                                                    beta=0,
                                                    bias=True,
                                                    **kwargs)


class LRPAlpha1Beta0IgnoreBias(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta0IgnoreBias, self).__init__(model, *args,
                                                            alpha=1,
                                                            beta=0,
                                                            bias=False,
                                                            **kwargs)
