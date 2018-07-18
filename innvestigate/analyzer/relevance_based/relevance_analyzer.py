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
import innvestigate.utils.keras as kutils
from innvestigate.utils.keras import checks as kchecks
from innvestigate.utils.keras import graph as kgraph
from . import relevance_rule as rrule
from . import utils as rutils



__all__ = [
    "BaselineLRPZ",

    "LRP",
    "LRP_RULES",

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

    "LRPSequentialPresetA",
    "LRPSequentialPresetB",

    "LRPSequentialPresetAFlat",
    "LRPSequentialPresetBFlat",

    "DeepTaylor",
    "BoundedDeepTaylor",
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

# Utility list enabling name mappings via string
LRP_RULES = {
    "Z": rrule.ZRule,
    "ZIgnoreBias": rrule.ZIgnoreBiasRule,

    "Epsilon": rrule.EpsilonRule,
    "EpsilonIgnoreBias": rrule.EpsilonIgnoreBiasRule,

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

class LRP(base.ReverseAnalyzerBase):
    """
    Base class for LRP-based model analyzers


    :param model: A Keras model.

    :param rule: A rule can be a  string or a Rule object, lists thereof or a list of conditions [(Condition, Rule), ... ]
      gradient.

    :param input_layer_rule: either a Rule object, atuple of (low, high) the min/max pixel values of the inputs
    """

    def __init__(self, model, *args, **kwargs):
        rule = kwargs.pop("rule", None)
        input_layer_rule = kwargs.pop("input_layer_rule", None)
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

                class BoundedProxyRule(rrule.BoundedRule):
                    def __init__(self, *args, **kwargs):
                        super(BoundedProxyRule, self).__init__(
                            *args, low=low, high=high, **kwargs)
                input_layer_rule = BoundedProxyRule


            if use_conditions is True:
                rules.insert(0,
                             (lambda layer, foo: kchecks.is_input_layer(layer),
                              input_layer_rule))

            else:
                rules.insert(0, input_layer_rule)





        ####################################################################
        ### Functionality responible for backwards rule selection below ####
        ####################################################################

        def select_rule(layer, reverse_state):
            ##print("in select_rule:", layer.__class__.__name__ , end='->') #debug
            if use_conditions is True:
                for condition, rule in rules:
                    if condition(layer, reverse_state):
                        ##print(str(rule)) #debug
                        return rule
                raise Exception("No rule applies to layer: %s" % layer)
            else:
                ##print(str(rules[0]), '(via pop)') #debug
                return rules.pop()


        # default backward hook
        class ReverseLayer(kgraph.ReverseMappingBase):
            def __init__(self, layer, state):
                rule_class = select_rule(layer, state) #NOTE: this prevents refactoring.
                ##print("in ReverseLayer.init:",layer.__class__.__name__,"->" , rule_class if isinstance(rule_class, six.string_types) else rule_class.__name__) #debug
                if isinstance(rule_class, six.string_types):
                    rule_class = LRP_RULES[rule_class]
                self._rule = rule_class(layer, state)

            def apply(self, Xs, Ys, Rs, reverse_state):
                ##print("    in ReverseLayer.apply:", reverse_state['layer'].__class__.__name__, '(nid: {})'.format(reverse_state['nid']) ,  '-> {}.apply'.format(self._rule.__class__.__name__))
                return self._rule.apply(Xs, Ys, Rs, reverse_state)


        #specialized backward hooks. TODO: add ReverseLayer class handling layers Without kernel: Add and AvgPool
        class BatchNormalizationReverseLayer(kgraph.ReverseMappingBase):
            def __init__(self, layer, state):
                ##print("in BatchNormalizationReverseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
                config = layer.get_config()

                self._center = config['center']
                self._scale = config['scale']
                self._axis = config['axis']

                self._mean = layer.moving_mean
                self._std = layer.moving_variance
                if self._center:
                    self._beta = layer.beta

                #TODO: implement rule support. for BatchNormalization -> [BNEpsilon, BNAlphaBeta, BNIgnore]
                #super(BatchNormalizationReverseLayer, self).__init__(layer, state)
                # how to do this:
                # super.__init__ calls select_rule and sets a self._rule class
                # check if isinstance(self_rule, EpsiloneRule), then reroute
                # to BatchNormEpsilonRule. Not pretty, but should work.

            def apply(self, Xs, Ys, Rs, reverse_state):
                ##print("    in BatchNormalizationReverseLayer.apply:", reverse_state['layer'].__class__.__name__, '(nid: {})'.format(reverse_state['nid']))

                input_shape = [K.int_shape(x) for x in Xs]
                if len(input_shape) != 1:
                    #extend below lambda layers towards multiple parameters.
                    raise ValueError("BatchNormalizationReverseLayer expects Xs with len(Xs) = 1, but was len(Xs) = {}".format(len(Xs)))
                input_shape = input_shape[0]

                # prepare broadcasting shape for layer parameters
                broadcast_shape = [1] * len(input_shape)
                broadcast_shape[self._axis] = input_shape[self._axis]
                broadcast_shape[0] =  -1

                #reweight relevances as
                #        x * (y - beta)     R
                # Rin = ---------------- * ----
                #           x - mu          y
                # batch norm can be considered as 3 distinct layers of subtraction,
                # multiplication and then addition. The multiplicative scaling layer
                # has no effect on LRP and functions as a linear activation layer

                minus_mu = keras.layers.Lambda(lambda x: x - K.reshape(self._mean, broadcast_shape))
                minus_beta = keras.layers.Lambda(lambda x: x - K.reshape(self._beta, broadcast_shape))
                prepare_div = keras.layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x,0), K.floatx())*2-1)*K.epsilon())


                x_minus_mu = kutils.apply(minus_mu, Xs)
                if self._center:
                    y_minus_beta = kutils.apply(minus_beta, Ys)
                else:
                    y_minus_beta = Ys

                numerator = [keras.layers.Multiply()([x, ymb, r])
                             for x, ymb, r in zip(Xs, y_minus_beta, Rs)]
                denominator = [keras.layers.Multiply()([xmm, y])
                             for xmm, y in zip(x_minus_mu, Ys)]

                return [ilayers.SafeDivide()([n, prepare_div(d)])
                        for n, d in zip(numerator, denominator)]

        class AddReverseLayer(kgraph.ReverseMappingBase):
            def __init__(self, layer, state):
                ##print("in AddReverseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
                self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                                     name_template="reversed_kernel_%s")

                #TODO: implement rule support.
                #super(AddReverseLayer, self).__init__(layer, state)

            def apply(self, Xs, Ys, Rs, reverse_state):
                # the outputs of the pooling operation at each location is the sum of its inputs.
                # the forward message must be known in this case, and are the inputs for each pooling thing.
                # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
                # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt
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



        class AveragePoolingRerseLayer(kgraph.ReverseMappingBase):
            def __init__(self, layer, state):
                ##print("in AveragePoolingRerseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
                self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                                     name_template="reversed_kernel_%s")

                #TODO: implement rule support.
                #super(AveragePoolingRerseLayer, self).__init__(layer, state)

            def apply(self, Xs, Ys, Rs, reverse_state):
                # the outputs of the pooling operation at each location is the sum of its inputs.
                # the forward message must be known in this case, and are the inputs for each pooling thing.
                # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
                # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt

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





        # conditional mappings layer_criterion -> ReverseLayer on how to handle backward passes through layers.
        self._conditional_mappings = [
            (kchecks.contains_kernel, ReverseLayer),
            (kchecks.is_batch_normalization_layer, BatchNormalizationReverseLayer),
            (kchecks.is_average_pooling, AveragePoolingRerseLayer),
            (kchecks.is_add_layer, AddReverseLayer),
        ]

        # FINALIZED constructor.
        super(LRP, self).__init__(model, *args, **kwargs)



    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        ##print("    in _default_reverse_mapping:", reverse_state['layer'].__class__.__name__, '(nid: {})'.format(reverse_state['nid']),  end='->')
        default_return_layers = [keras.layers.Activation]# TODO extend
        if(len(Xs) == len(Ys) and
           isinstance(reverse_state['layer'], (keras.layers.Activation,)) and
           all([K.int_shape(x) == K.int_shape(y) for x, y in zip(Xs, Ys)])):
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            ##print('return R')
            return reversed_Ys
        else:
            # This branch covers:
            # MaxPooling
            # Average Pooling
            # Max
            # Flatten
            # Reshape
            # Concatenate
            # Cropping
            ##print('ilayers.GradientWRT')
            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)


    ########################################
    ### End of Rule Selection Business. ####
    ########################################


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

        class EpsilonProxyRule(rrule.EpsilonRule):
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


class LRPEpsilonIgnoreBias(LRPEpsilon):

    def __init__(self, model, epsilon=1e-7, *args, **kwargs):
        super(LRPEpsilonIgnoreBias, self).__init__(model, *args,
                                                   epsilon=epsilon,
                                                   bias=False,
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

        class AlphaBetaProxyRule(rrule.AlphaBetaRule):
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


class LRPSequentialPresetA(_LRPFixedParams): #for the lack of a better name
    def __init__(self, model, epsilon=1e-1, *args, **kwargs):
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
                        "networks with non-ReLU activations.") #TODO: fix. specify. extend.
        }
        ]

        class EpsilonProxyRule(rrule.EpsilonRule):
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=epsilon,
                                                       bias=True,
                                                       **kwargs)


        conditional_rules = [(kchecks.is_dense_layer, EpsilonProxyRule),
                             (kchecks.is_conv_layer, rrule.Alpha1Beta0Rule)
                            ]

        super(LRPSequentialPresetA, self).__init__(model,
                                            *args,
                                            rule = conditional_rules,
                                            **kwargs )


class LRPSequentialPresetB(_LRPFixedParams):
    def __init__(self, model, epsilon=1e-1, *args, **kwargs):
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
                        "networks with non-ReLU activations.") #TODO: fix. specify. extend.
        }
        ]

        class EpsilonProxyRule(rrule.EpsilonRule):
            def __init__(self, *args, **kwargs):
                super(EpsilonProxyRule, self).__init__(*args,
                                                       epsilon=epsilon,
                                                       bias=True,
                                                       **kwargs)


        conditional_rules = [(kchecks.is_dense_layer, EpsilonProxyRule),
                             (kchecks.is_conv_layer, rrule.Alpha2Beta1Rule)
                            ]
        super(LRPSequentialPresetB, self).__init__(model,
                                            *args,
                                            rule = conditional_rules,
                                            **kwargs )





#TODO: allow to pass input layer identification by index or id.
class LRPSequentialPresetAFlat(LRPSequentialPresetA):
    def __init__(self, model, *args, **kwargs):
        super(LRPSequentialPresetAFlat, self).__init__(model,
                                                *args,
                                                input_layer_rule=rrule.FlatRule,
                                                **kwargs)



#TODO: allow to pass input layer identification by index or id.
class LRPSequentialPresetBFlat(LRPSequentialPresetB):
    def __init__(self, model, *args, **kwargs):
        super(LRPSequentialPresetBFlat, self).__init__(model,
                                                *args,
                                                input_layer_rule="Flat",
                                                **kwargs)


class DeepTaylor(LRPAlpha1Beta0):

    def __init__(self, model, *args, **kwargs):

        # TODO(ALBER) This code is mostly copied and should be refactored.
        class DeepTaylorAveragePoolingRerseLayer(kgraph.ReverseMappingBase):
            def __init__(self, layer, state):
                if isinstance(layer, keras.layers.pooling.MaxPooling1D):
                    layer_replacement = keras.layers.pooling.AveragePooling1D(
                        pool_size=layer.pool_size, strides=layer.strides,
                        padding=layer.padding)
                elif isinstance(layer, keras.layers.pooling.MaxPooling2D):
                    layer_replacement = keras.layers.pooling.AveragePooling2D(
                        pool_size=layer.pool_size, strides=layer.strides,
                        padding=layer.padding, data_format=layer.data_format)
                elif isinstance(layer, keras.layers.pooling.MaxPooling3D):
                    layer_replacement = keras.layers.pooling.AveragePooling3D(
                        pool_size=layer.pool_size, strides=layer.strides,
                        padding=layer.padding, data_format=layer.data_format)
                elif isinstance(layer, keras.layers.pooling.GlobalMaxPooling1D):
                    layer_replacement = keras.layers.pooling.GlobalAveragePooling1D()
                elif isinstance(layer, keras.layers.pooling.GlobalMaxPooling2D):
                    layer_replacement = keras.layers.pooling.GlobalAveragePooling2D(
                        data_format=layer.data_format)
                elif isinstance(layer, keras.layers.pooling.GlobalMaxPooling3D):
                    layer_replacement = keras.layers.pooling.GlobalAveragePooling3D(
                        data_format=layer.data_format)
                else:
                    raise Exception()

                self._layer_wo_act = layer_replacement

                #TODO: implement rule support.
                #super(AveragePoolingRerseLayer, self).__init__(layer, state)

            def apply(self, Xs, Ys, Rs, reverse_state):
                # the outputs of the pooling operation at each location is the sum of its inputs.
                # the forward message must be known in this case, and are the inputs for each pooling thing.
                # the gradient is 1 for each output-to-input connection, which corresponds to the "weights"
                # of the layer. It should thus be sufficient to reweight the relevances and and do a gradient_wrt

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

        super(DeepTaylor, self).__init__(model, *args, **kwargs)

        # TODO(ALBER) make this conditional more transparent

        self._conditional_mappings += [
            (kchecks.is_max_pooling, DeepTaylorAveragePoolingRerseLayer),
        ]

class BoundedDeepTaylor(DeepTaylor):

    def __init__(self, model, *args, low=None, high=None, **kwargs):

        if low is None or high is None:
            # TODO(ALBER) Put better error message.
            raise ValueError("The (low, high) value for the Z-B (bounded rule)"
                             " input rule must be specified.")

        class BoundedProxyRule(rrule.BoundedRule):
            def __init__(self, *args, **kwargs):
                super(BoundedProxyRule, self).__init__(
                    *args, low=low, high=high, **kwargs)

        super(BoundedDeepTaylor, self).__init__(
            model, *args, input_layer_rule=BoundedProxyRule, **kwargs)
