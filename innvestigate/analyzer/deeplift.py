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
import keras.layers

import numpy as np

from . import base
from .. import utils as iutils
from .. import layers as ilayers

from ..utils import keras as kutils

from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph




__all__ = [
    "DeepLIFTCore",
]


###############################################################################
###############################################################################
###############################################################################

class DeepLIFTCore(base.ReverseAnalyzerBase):

    def __init__(self, model, reference_inputs=0, *args, **kwargs):
        self._reference_inputs = reference_inputs
        self._add_model_softmax_check()
        super(DeepLIFTCore, self).__init__(model, *args, **kwargs)

    def _prepare_model(self, model):
        ret = super(DeepLIFTCore, self)._prepare_model(model)
        # Store analysis input to create reference inputs.
        self._analysis_inputs = ret[1]
        return ret

    def _create_reference_activations(self, model):
        # TODO(albermax): cache this information once new backend code is
        # merged.
        trace = kgraph.trace_model_execution(model)
        layers, execution_list, outputs = trace

        self._reference_activations = {}

        # Create and inputs.
        # TODO(albermax): remove this function once new backend code is
        # merged.
        def broadcast_np_tensors_to_keras_tensors(keras_tensors, np_tensors):
            def none_to_one(tmp):
                return [1 if x is None else x for x in tmp]

            keras_tensors = iutils.to_list(keras_tensors)

            if isinstance(np_tensors, list):
                ret = [np.broadcast_to(ri, none_to_one(K.int_shape(x)))
                       for x, ri in zip(keras_tensors, np_tensors)]
            else:
                ret = [np.broadcast_to(np_tensors,
                                       none_to_one(K.int_shape(x)))
                       for x in keras_tensors]

            return ret

        tmp = broadcast_np_tensors_to_keras_tensors(
            model.inputs, self._reference_inputs)
        tmp = [K.variable(x) for x in tmp]

        constant_reference_inputs = [
            keras.layers.Input(tensor=x, shape=K.int_shape(x)[1:])
            for x in tmp
        ]

        for k, v in zip(model.inputs, constant_reference_inputs):
            self._reference_activations[k] = v

        for k, v in zip(self._analysis_inputs, self._analysis_inputs):
            self._reference_activations[k] = v

        # Compute intermediate states.
        for layer, Xs, Ys in execution_list:
            activations = [self._reference_activations[x] for x in Xs]

            if isinstance(layer, keras.layers.InputLayer):
                # Special case. Do nothing.
                next_activations = activations
            else:
                next_activations = iutils.to_list(
                    kutils.apply(layer, activations))

            assert len(next_activations) == len(Ys)
            for k, v in zip(Ys, next_activations):
                self._reference_activations[k] = v

        return constant_reference_inputs

    def _create_analysis(self, model, *args, **kwargs):
        constant_reference_inputs = self._create_reference_activations(model)

        for l in self._model.layers:
            for k in iutils.to_list(l.get_input_at(0)):
                if k in self._reference_activations:
                    l.reference_activation = self._reference_activations[k]

        self._add_conditional_reverse_mapping(
            lambda l: (not kchecks.contains_kernel(l) and
                       kchecks.contains_activation(l)),
            DeepLIFTRescaleRule,
            name="deeplift_activation_layer",
        )

        self._add_conditional_reverse_mapping(
            lambda l: kchecks.contains_kernel(l),
            DeepLIFTLinearRule,
            name="deeplift_kernel_layer",
        )

        tmp = super(DeepLIFTCore, self)._create_analysis(
            model, *args, **kwargs)

        if isinstance(tmp, tuple):
            if len(tmp) == 3:
                analysis_outputs, debug_outputs, constant_inputs = tmp
            elif len(tmp) == 2:
                analysis_outputs, debug_outputs = tmp
                constant_inputs = list()
            elif len(tmp) == 1:
                analysis_outputs = iutils.to_list(tmp[0])
                constant_inputs, debug_outputs = list(), list()
            else:
                raise Exception("Unexpected output from _create_analysis.")
        else:
            analysis_outputs = tmp
            constant_inputs, debug_outputs = list(), list()

        return (analysis_outputs,
                debug_outputs,
                constant_inputs + constant_reference_inputs)

    def _head_mapping(self, X):
        return keras.layers.Subtract()([X, self._reference_activations[X]])

    def _get_state(self):
        state = super(DeepLIFTCore, self)._get_state()
        state.update({"reference_inputs": self._reference_inputs})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        reference_inputs = state.pop("reference_inputs")
        kwargs = super(DeepLIFTCore, clazz)._state_to_kwargs(state)
        kwargs.update({"reference_inputs": reference_inputs})
        return kwarg


class DeepLIFTLinearRule(kgraph.ReverseMappingBase):
    """
    Basic DeepLiftRule decomposition rule (for layers with weight kernels),
    which considers the bias a constant input neuron.
    """

    def __init__(self, layer, state, bias=True):

        # Copy forward layer, but without activations
        self._layer = kgraph.copy_layer_wo_activation(layer)

        self.ref_Xs = layer.reference_activation
        self.ref_Zs = kutils.apply(self._layer, [self.ref_Xs])[0]

        # If this layer has an activation, we need to do apply a non-linear
        # rule first (Rescale or RevealCancel) before applying the linear rule
        # x -> z -> f(.) -> y
        if (kchecks.contains_activation(layer)):
            self._activation = layer.activation
            self._nonlinear_rule = DeepLIFTRescaleRule(layer.activation, state)
            self._nonlinear_rule.ref_Xs = self.ref_Zs

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        Zs = kutils.apply(self._layer, Xs)
        if self._activation:
            Rs = self._nonlinear_rule.apply(Zs, Ys, Rs, reverse_state)

        delta_Zs = [keras.layers.Lambda(lambda z: z - self.ref_Zs)(z) for z in Zs]
        delta_Xs = [keras.layers.Lambda(lambda x: x - self.ref_Xs)(x) for x in Xs]

        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, delta_Zs)]

        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs + Zs + tmp))

        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(delta_Xs, tmp)]


class DeepLIFTRescaleRule(kgraph.ReverseMappingBase):
    def __init__(self, activation_layer, state, bias=True):
        # x -> f(.) -> y
        self._layer = activation_layer

    def apply(self, Xs, Ys, Rs, reverse_state):
        Ys = [keras.layers.Lambda(lambda x: self._layer(x))(x) for x in Xs]

        ref_Ys = kutils.apply(self._layer, self.ref_Xs)

        delta_Xs = [keras.layers.Lambda(lambda x: x - self.ref_Xs)(x) for x in Xs]
        delta_Ys = [keras.layers.Lambda(lambda y: y - ref_Ys)(y) for y in Ys]
        is_delta_Xs_zero = [keras.layers.Lambda(lambda x: K.cast(K.less(x, K.epsilon()), K.dtype(x)))(dx) for dx in
                            delta_Xs]

        grad = iutils.to_list(ilayers.GradientWRT(len(Xs))(Xs + Ys + Rs))

        multipliers = []
        for dy, dx, m, g in zip(delta_Ys, delta_Xs, is_delta_Xs_zero, grad):
            normal_path = keras.layers.Multiply()([
                keras.layers.Lambda(lambda x: 1 - x)(m),
                ilayers.SafeDivide()([dy, dx])
            ])

            grad_path = keras.layers.Multiply()([m, g])

            mul = keras.layers.Add()([normal_path, grad_path])

            multipliers.append(mul)

        return [keras.layers.Multiply()([a, b])
                for a, b in zip(multipliers, Rs)]

