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


import keras.backend as K
import keras.models
import keras
import numpy as np


from . import base
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import graph as kgraph

__all__ = [
    "BaselineGradient",
    "Gradient",

    "Deconvnet",
    "GuidedBackprop",

    "IntegratedGradients",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineGradient(base.AnalyzerNetworkBase):

    properties = {
        "name": "BaselineGradient",
        "show_as": "rgb",
    }

    def _create_analysis(self, model):
        return ilayers.Gradient()(model.inputs+[model.outputs[0], ])


class Gradient(base.ReverseAnalyzerBase):

    properties = {
        "name": "Gradient",
        "show_as": "rgb",
    }

    def _head_mapping(self, X):
        return ilayers.OnesLike()(X)


###############################################################################
###############################################################################
###############################################################################


class Deconvnet(base.ReverseAnalyzerBase):

    properties = {
        "name": "Deconvnet",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_relu_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "Deconvnet is only well defined for "
            "convluational neural networks with non-relu activations."
            )

        class ReverseLayer(kgraph.ReverseMappingBase):

            def __init__(self, layer, state):
                self._activation = keras.layers.Activation("relu")
                self._layer_wo_relu = kgraph.get_layer_wo_activation(
                    layer,
                    name_template="reversed_%s",
                )

            def apply(self, Xs, Ys, reversed_Ys, reverse_state):
                # apply relus conditioned on backpropagated values.
                reversed_Ys = kutils.easy_apply(self._activation, reversed_Ys)

                # apply gradient of forward without relus
                Ys_wo_relu = kutils.easy_apply(self._layer_wo_relu, Xs)
                return ilayers.GradientWRT(len(Xs))(Xs+Ys_wo_relu+reversed_Ys)

        # todo: add check for other non-linearities.
        self._conditional_mappings = [
            (lambda layer: kgraph.contains_activation(layer, "relu"),
             ReverseLayer),
        ]
        return super(Deconvnet, self).__init__(*args, **kwargs)


class GuidedBackprop(base.ReverseAnalyzerBase):

    properties = {
        "name": "GuidedBackprop",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_relu_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "GuidedBackprop is only well defined for "
            "convluational neural networks with non-relu activations."
            )

        def reverse_layer_instance(Xs, Ys, reversed_Ys, reverse_state):
            activation = keras.layers.Activation("relu")
            reversed_Ys = kutils.easy_apply(activation, reversed_Ys)

            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

        # todo: add check for other non-linearities.
        self._conditional_mappings = [
            (lambda layer: kgraph.contains_activation(layer, "relu"),
             reverse_layer_instance),
        ]
        return super(GuidedBackprop, self).__init__(*args, **kwargs)


###############################################################################
###############################################################################
###############################################################################


class IntegratedGradients(Gradient):

    properties = {
        "name": "IntegratedGradients",
        "show_as": "rgb",
    }

    def __init__(self, model, *args, steps=16, reference_inputs=0, **kwargs):
        self._steps = steps
        self._reference_inputs = reference_inputs

        return super(IntegratedGradients, self).__init__(model,
                                                         *args, **kwargs)

    def _create_analysis(self, model):
        def none_to_one(l):
            return [1 if x is None else x for x in l]

        if isinstance(self._reference_inputs, list):
            reference_inputs = [np.broadcast_to(x,
                                                none_to_one(K.int_shape(tmp)))
                                for x, tmp in zip(self._reference_inputs,
                                                  model.inputs)]
        else:
            reference_inputs = [np.broadcast_to(self._reference_inputs,
                                                none_to_one(K.int_shape(tmp)))
                                for tmp in model.inputs]

        reference_inputs = [
            keras.layers.Input(tensor=K.variable(x), shape=x.shape)
            for x in reference_inputs
        ]

        difference = [ilayers.Sum()([x, reference])
                      for x, reference in zip(model.inputs, reference_inputs)]

        def augment_input(x, difference):
            ret = []
            shape = K.int_shape(x)
            for i in range(self._steps):
                scale = keras.layers.Lambda(lambda x: 1.0*i/self._steps * x)
                tmp = keras.layers.Add()([x, scale(difference)])
                first_dimension = -1 if shape[0] is None else shape[0]
                tmp = keras.layers.Reshape((first_dimension, 1)+shape[1:])(tmp)
                ret.append(tmp)
            ret = keras.layers.Concatenate(axis=1)(ret)
            first_dimension = -1 if shape[0] is None else shape[0]*self._steps
            ret = keras.layers.Reshape((first_dimension,)+shape[1:])(ret)
            return ret

        def reduce_output(x):
            tmp = keras.layers.Reshape(shape=(steps, xxx))(x)
            #sum along first axis
            #multiply with difference and 1/m
            return x

        augmented_inputs = [augment_input(x, d)
                            for x, d in zip(model.inputs, reference_inputs)]

        augmented_model = keras.models.Model(
            inputs=model.inputs+reference_inputs,
            outputs=model(augmented_inputs))

        gradients = super(IntegratedGradients,
                          self)._create_analysis(augmented_model)

        return ([reduce_output(x, d)
                 for x, d in zip(iutils.listify(gradients), reference_inputs)],
                list(),
                reference_inputs)

    def _get_state(self):
        state = super(IntegratedGradients, self)._get_state()
        state.update({"steps": self._steps})
        state.update({"reference_inputs": self._reference_inputs})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        steps = state.pop("steps")
        reference_input = state.pop("reference_input")
        kwargs = super(IntegratedGradients, clazz)._state_to_kwargs(state)
        kwargs.update({"reference_inputs": reference_inputs})
        return kwargs
