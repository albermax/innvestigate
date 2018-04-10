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


from .. import base
from innvestigate import layers as ilayers
from innvestigate import utils as iutils
from innvestigate.utils.keras import checks as kchecks
from innvestigate.utils.keras import graph as kgraph
from . import utils as rutils


__all__ = [
    "BaselineLRPZ",

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
    "LRPZPlusFast",

    "LRPCompositeA",
    "LRPCompositeB",

    "LRPCompositeAFlat",
    "LRPCompositeBFlat",

    "LRPCompositeAWSquare",
    "LRPCompositeBWSquare",
]


###############################################################################
###############################################################################
###############################################################################

class BaselineLRPZ(base.AnalyzerNetworkBase):
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
            # the given rule is a single string or single rule implementing cla ss
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


        # create a BoundedRule for input layer handling from given tuple
        if self._input_layer_rule is not None:
            input_layer_rule = self._input_layer_rule
            if isinstance(input_layer_rule, tuple):
                low, high = input_layer_rule

                class input_layer_rule(BoundedRule):
                    def __init__(self, *args, **kwargs):
                        super(input_layer_rule, self).__init__(
                            *args, low=low, high=high, **kwargs)


            if use_conditions is True:
                rules.insert(0,
                             (lambda layer, foo: kchecks.is_input_layer(layer),
                              input_layer_rule))

            else:
                rules.insert(0, input_layer_rule)


        def select_rule(layer, reverse_state): #TODO make module fxn.
            # TODO: check if use_conditions functions properly. should be class variable.
            #print(layer.__class__.__name__, end='->') #debug
            if use_conditions is True:
                for condition, rule in rules:
                    if condition(layer, reverse_state):
                        #print(str(rule)) #debug
                        return rule
                raise Exception("No rule applies to layer: %s" % layer)
            else:
                #print(str(rules[0]) + ' (pop)') #debug
                return rules.pop()


        class ReverseLayer(kgraph.ReverseMappingBase):
            # TODO: refactor as independent class?
            def __init__(self, layer, state):
                rule_class = select_rule(layer, state) #this avoids refactoring.
                #print(layer, rule_class) #debug
                if isinstance(rule_class, six.string_types):
                    rule_class = LRP_RULES[rule_class]
                self._rule = rule_class(layer, state)

            def apply(self, Xs, Ys, Rs, reverse_state):
                #print(reverse_state['layer'].__class__.__name__, 'Rule.apply kicking in for rule', self._rule)
                return self._rule.apply(Xs, Ys, Rs, reverse_state)


        # conditional mappings layer_criterion -> Rule on how to handle backward passes through layers.
        self._conditional_mappings = [
            (kchecks.contains_kernel, ReverseLayer),
            #TODO: BatchNOrm, MaxPooling (ReverseLayer), SumPooling (ReverseLayer), Flatten, Reshape
        ]

        # finalize constructor.
        super(LRP, self).__init__(model, *args, **kwargs)




    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        #print(reverse_state['layer'].__class__.__name__, '_default_reverse_layer', end=':')
        default_return_layers = [keras.layers.Activation]# TODO extend
        if(len(Xs) == len(Ys) and
           all([K.int_shape(x) == K.int_shape(y) for x, y in zip(Xs, Ys)])):
        #if isinstance(reverse_state['layer'], keras.layers.Activation): # TODO: complete this. Activation should not be everything.
            # TODO: this is not necessarily true. Do explicit layer check.
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            #print(' return R')
            return reversed_Ys
        else:
            # TODO: make this more clear, here we assume to have reshape layers
            # TODO: add assert
            # TODO: BatchNorm layer should end up here (?): Implements an affine transformation.
            # TODO: Confirm that behaviour of GradientWRT. Flatten and BatchNorm are correct
            #print(' ilayers.GradientWRT')
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
# ANALYZER CLASSES AND PRESETS ################################################
###############################################################################


class _LRPFixedParams(LRP):

    @classmethod
    def _state_to_kwargs(clazz, state):
        kwargs = super(_LRPFixedParams, clazz)._state_to_kwargs(state)
        del kwargs["rule"]
        return kwargs


class LRPZ(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPZ, self).__init__(model, *args, rule="Z", **kwargs)


class LRPZIgnoreBias(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPZIgnoreBias, self).__init__(model, *args,
                                             rule="ZIgnoreBias", **kwargs)



class LRPEpsilon(_LRPFixedParams):

    def __init__(self, model, epsilon=1e-7, bias=True, *args, **kwargs):
        epsilon = rutils.assert_lrp_epsilon_param(epsilon, self)
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

        super(LRPEpsilon, self).__init__(model, *args,
                                         rule=EpsilonProxyRule,
                                         **kwargs)


class LRPEpsilonIgnoreBias(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPEpsilonIgnoreBias, self).__init__(model, *args,
                                                   rule="EpsilonIgnoreBias",
                                                   **kwargs)


class LRPWSquare(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPWSquare, self).__init__(model, *args,
                                         rule="WSquare", **kwargs)


class LRPFlat(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPFlat, self).__init__(model, *args,
                                      rule="Flat", **kwargs)


class LRPAlphaBeta(LRP):
    """ Base class for LRP AlphaBeta"""

    def __init__(self, model, alpha=None, beta=None, bias=True, *args, **kwargs):
        alpha, beta = rutils.assert_infer_lrp_alpha_beta_param(alpha, beta, self)
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

        super(LRPAlphaBeta, self).__init__(model, *args,
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


class LRPAlpha2Beta1(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha2Beta1, self).__init__(model, *args,
                                             alpha=2,
                                             beta=1,
                                             bias=True,
                                             **kwargs)


class LRPAlpha2Beta1IgnoreBias(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha2Beta1IgnoreBias, self).__init__(model, *args,
                                                       alpha=2,
                                                       beta=1,
                                                       bias=False,
                                                       **kwargs)


class LRPAlpha1Beta0(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha1Beta0, self).__init__(model, *args,
                                             alpha=1,
                                             beta=0,
                                             bias=True,
                                             **kwargs)


class LRPAlpha1Beta0IgnoreBias(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        super(LRPAlpha1Beta0IgnoreBias, self).__init__(model, *args,
                                                       alpha=1,
                                                       beta=0,
                                                       bias=False,
                                                       **kwargs)

class LRPZPlus(LRPAlpha1Beta0IgnoreBias):
    #TODO: assert that layer inputs are always >= 0
    def __init__(self, model, *args, **kwargs):
        super(LRPZPlus, self).__init__(model, *args, **kwargs)


class LRPZPlusFast(_LRPFixedParams):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """
    #TODO: assert that layer inputs are always >= 0
    def __init__(self, model, *args, **kwargs):
        super(LRPZPlusFast, self).__init__(model, *args,
                                       rule="ZPlusFast", **kwargs)


class LRPCompositeA(_LRPFixedParams): #for the lack of a better name
    def __init__(self, model, *args, **kwargs):
        self._model_checks = [
        # TODO: Check for non-linear output in general.
        {
            "check": lambda layer: kchecks.contains_activation(
                layer, activation="softmax"),
            "type": "exception",
            "message": "Model should not contain a softmax.",
        },
        {
            "check":
            lambda layer: not kchecks.only_relu_activation(layer),
            "type": "warning",
            "message": (" is not advised for "
                        "networks with non-ReLU activations.")
        }
        ]

        class EpsilonProxyRule(EpsilonRule):
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=1e-3,
                                                       bias=True,
                                                       **kwargs)


        conditional_rules = [(kchecks.is_dense_layer, EpsilonProxyRule),
                             (kchecks.is_conv_layer, Alpha1Beta0Rule)
                            ]

        super(LRPCompositeA, self).__init__(model,
                                           *args,
                                           rule = conditional_rules,
                                           **kwargs )


class LRPCompositeB(_LRPFixedParams):
    def __init__(self, model, *args, **kwargs):
        self._model_checks = [
        # TODO: Check for non-linear output in general.
        {
            "check": lambda layer: kchecks.contains_activation(
                layer, activation="softmax"),
            "type": "exception",
            "message": "Model should not contain a softmax.",
        },
        {
            "check":
            lambda layer: not kchecks.only_relu_activation(layer),
            "type": "warning",
            "message": (" is not advised for "
                        "networks with non-ReLU activations.")
        }
        ]

        class EpsilonProxyRule(EpsilonRule):
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=1e-3,
                                                       bias=True,
                                                       **kwargs)


        conditional_rules = [(kchecks.is_dense_layer, EpsilonProxyRule),
                             (kchecks.is_conv_layer, Alpha2Beta1Rule)
                            ]
        super(LRPCompositeB, self).__init__(model,
                                           *args,
                                           rule = conditional_rules,
                                           **kwargs )





#TODO: allow to pass input layer identification by index or id.
class LRPCompositeAFlat(LRPCompositeA):
    def __init__(self, model, *args, **kwargs):
        super(LRPCompositeAFlat, self).__init__(model,
                                                *args,
                                                input_layer_rule=FlatRule,
                                                **kwargs)



#TODO: allow to pass input layer identification by index or id.
class LRPCompositeBFlat(LRPCompositeB):
    def __init__(self, model, *args, **kwargs):
        super(LRPCompositeBFlat, self).__init__(model,
                                                *args,
                                                input_layer_rule="Flat",
                                                **kwargs)


#TODO: allow to pass input layer identification by index or id.
class LRPCompositeAWSquare(LRPCompositeA):
    def __init__(self, model, *args, **kwargs):
        super(LRPCompositeAWSquare, self).__init__(model,
                                                *args,
                                                input_layer_rule=WSquareRule,
                                                **kwargs)



#TODO: allow to pass input layer identification by index or id.
class LRPCompositeBWSquare(LRPCompositeB):
    def __init__(self, model, *args, **kwargs):
        super(LRPCompositeBWSquare, self).__init__(model,
                                                *args,
                                                input_layer_rule="WSquare",
                                                **kwargs)
