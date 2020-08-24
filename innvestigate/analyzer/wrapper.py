# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import zip


###############################################################################
###############################################################################
###############################################################################


import keras.models
import keras.backend as K
import numpy as np


from . import base
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils


__all__ = [
    "WrapperBase",
    "AugmentReduceBase",
    "GaussianSmoother",
    "PathIntegrator",
]


###############################################################################
###############################################################################
###############################################################################


class WrapperBase(base.AnalyzerBase):
    """Interface for wrappers around analyzers

    This class is the basic interface for wrappers around analyzers.

    :param subanalyzer: The analyzer to be wrapped.
    """

    def __init__(self, subanalyzer, *args, **kwargs):
        self._subanalyzer = subanalyzer
        model = None

        super(WrapperBase, self).__init__(model,
                                          *args, **kwargs)

    def analyze(self, *args, **kwargs):
        return self._subanalyzer.analyze(*args, **kwargs)

    def _get_state(self):
        sa_class_name, sa_state = self._subanalyzer.save()

        state = {}
        state.update({"subanalyzer_class_name": sa_class_name})
        state.update({"subanalyzer_state": sa_state})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        sa_class_name = state.pop("subanalyzer_class_name")
        sa_state = state.pop("subanalyzer_state")
        assert len(state) == 0

        subanalyzer = base.AnalyzerBase.load(sa_class_name, sa_state)
        kwargs = {"subanalyzer": subanalyzer}
        return kwargs


###############################################################################
###############################################################################
###############################################################################


class AugmentReduceBase(WrapperBase):
    """Interface for wrappers that augment the input and reduce the analysis.

    This class is an interface for wrappers that:
    * augment the input to the analyzer by creating new samples.
    * reduce the returned analysis to match the initial input shapes.

    :param subanalyzer: The analyzer to be wrapped.
    :param augment_by_n: Number of samples to create.
    """

    def __init__(self, subanalyzer, *args, **kwargs):
        self._augment_by_n = kwargs.pop("augment_by_n", 2)
        self._neuron_selection_mode = subanalyzer._neuron_selection_mode

        if self._neuron_selection_mode != "all":
            # TODO: this is not transparent, find a better way.
            subanalyzer._neuron_selection_mode = "index"
        super(AugmentReduceBase, self).__init__(subanalyzer,
                                                *args, **kwargs)

        if isinstance(self._subanalyzer, base.AnalyzerNetworkBase):
            # Take the keras analyzer model and
            # add augment and reduce functionality.
            self._keras_based_augment_reduce = True
        else:
            raise NotImplementedError("Keras-based subanalyzer required.")

    def create_analyzer_model(self):
        if not self._keras_based_augment_reduce:
            return

        self._subanalyzer.create_analyzer_model()

        if self._subanalyzer._n_debug_output > 0:
            raise Exception("No debug output at subanalyzer is supported.")

        model = self._subanalyzer._analyzer_model
        if None in model.input_shape[1:]:
            raise ValueError("The input shape for the model needs "
                             "to be fully specified (except the batch axis). "
                             "Model input shape is: %s" % (model.input_shape,))

        inputs = model.inputs[:self._subanalyzer._n_data_input]
        extra_inputs = model.inputs[self._subanalyzer._n_data_input:]
        # todo: check this, index seems not right.
        #outputs = model.outputs[:self._subanalyzer._n_data_input]
        extra_outputs = model.outputs[self._subanalyzer._n_data_input:]

        if len(extra_outputs) > 0:
            raise Exception("No extra output is allowed "
                            "with this wrapper.")

        new_inputs = iutils.to_list(self._augment(inputs))
        # print(type(new_inputs), type(extra_inputs))
        tmp = iutils.to_list(model(new_inputs+extra_inputs))
        new_outputs = iutils.to_list(self._reduce(tmp))
        new_constant_inputs = self._keras_get_constant_inputs()

        new_model = keras.models.Model(
            inputs=inputs+extra_inputs+new_constant_inputs,
            outputs=new_outputs+extra_outputs)
        self._subanalyzer._analyzer_model = new_model

    def analyze(self, X, *args, **kwargs):
        if self._keras_based_augment_reduce is True:
            if not hasattr(self._subanalyzer, "_analyzer_model"):
                self.create_analyzer_model()

            ns_mode = self._neuron_selection_mode
            if ns_mode in ["max_activation", "index"]:
                if ns_mode == "max_activation":
                    tmp = self._subanalyzer._model.predict(X)
                    indices = np.argmax(tmp, axis=1)
                else:
                    if len(args):
                        args = list(args)
                        indices = args.pop(0)
                    else:
                        indices = kwargs.pop("neuron_selection")

                # broadcast to match augmented samples.
                indices = np.repeat(indices, self._augment_by_n)

                kwargs["neuron_selection"] = indices
            return self._subanalyzer.analyze(X, *args, **kwargs)
        else:
            raise DeprecationWarning("Not supported anymore.")

    def _keras_get_constant_inputs(self):
        return list()

    def _augment(self, X):
        repeat = ilayers.Repeat(self._augment_by_n, axis=0)
        return [repeat(x) for x in iutils.to_list(X)]

    def _reduce(self, X):
        X_shape = [K.int_shape(x) for x in iutils.to_list(X)]
        reshape = [ilayers.Reshape((-1, self._augment_by_n)+shape[1:])
                   for shape in X_shape]
        mean = ilayers.Mean(axis=1)

        return [mean(reshape_x(x)) for x, reshape_x in zip(X, reshape)]

    def _get_state(self):
        if self._neuron_selection_mode != "all":
            # TODO: this is not transparent, find a better way.
            # revert the tempering in __init__
            tmp = self._neuron_selection_mode
            self._subanalyzer._neuron_selection_mode = tmp
        state = super(AugmentReduceBase, self)._get_state()
        state.update({"augment_by_n": self._augment_by_n})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        augment_by_n = state.pop("augment_by_n")
        kwargs = super(AugmentReduceBase, clazz)._state_to_kwargs(state)
        kwargs.update({"augment_by_n": augment_by_n})
        return kwargs


###############################################################################
###############################################################################
###############################################################################


class GaussianSmoother(AugmentReduceBase):
    """Wrapper that adds noise to the input and averages over analyses

    This wrapper creates new samples by adding Gaussian noise
    to the input. The final analysis is an average of the returned analyses.

    :param subanalyzer: The analyzer to be wrapped.
    :param noise_scale: The stddev of the applied noise.
    :param augment_by_n: Number of samples to create.
    """

    def __init__(self, subanalyzer, *args, **kwargs):
        self._noise_scale = kwargs.pop("noise_scale", 1)
        super(GaussianSmoother, self).__init__(subanalyzer,
                                               *args, **kwargs)

    def _augment(self, X):
        tmp = super(GaussianSmoother, self)._augment(X)
        noise = ilayers.TestPhaseGaussianNoise(stddev=self._noise_scale)
        return [noise(x) for x in tmp]

    def _get_state(self):
        state = super(GaussianSmoother, self)._get_state()
        state.update({"noise_scale": self._noise_scale})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        noise_scale = state.pop("noise_scale")
        kwargs = super(GaussianSmoother, clazz)._state_to_kwargs(state)
        kwargs.update({"noise_scale": noise_scale})
        return kwargs


###############################################################################
###############################################################################
###############################################################################


class PathIntegrator(AugmentReduceBase):
    """Integrated the analysis along a path

    This analyzer:
    * creates a path from input to reference image.
    * creates steps number of intermediate inputs and
      crests an analysis for them.
    * sums the analyses and multiplies them with the input-reference_input.

    This wrapper is used to implement Integrated Gradients.
    We refer to the paper for further information.

    :param subanalyzer: The analyzer to be wrapped.
    :param steps: Number of steps for integration.
    :param reference_inputs: The reference input.
    """

    def __init__(self, subanalyzer, *args, **kwargs):
        steps = kwargs.pop("steps", 16)
        self._reference_inputs = kwargs.pop("reference_inputs", 0)
        self._keras_constant_inputs = None
        super(PathIntegrator, self).__init__(subanalyzer,
                                             *args,
                                             augment_by_n=steps,
                                             **kwargs)

    def _keras_set_constant_inputs(self, inputs):
        tmp = [K.variable(x) for x in inputs]
        self._keras_constant_inputs = [
            keras.layers.Input(tensor=x, shape=x.shape[1:])
            for x in tmp]

    def _keras_get_constant_inputs(self):
        return self._keras_constant_inputs

    def _compute_difference(self, X):
        if self._keras_constant_inputs is None:
            tmp = kutils.broadcast_np_tensors_to_keras_tensors(
                X, self._reference_inputs)
            self._keras_set_constant_inputs(tmp)

        reference_inputs = self._keras_get_constant_inputs()
        return [keras.layers.Subtract()([x, ri])
                for x, ri in zip(X, reference_inputs)]

    def _augment(self, X):
        tmp = super(PathIntegrator, self)._augment(X)
        tmp = [ilayers.Reshape((-1, self._augment_by_n)+K.int_shape(x)[1:])(x)
               for x in tmp]

        difference = self._compute_difference(X)
        self._keras_difference = difference
        # Make broadcastable.
        difference = [ilayers.Reshape((-1, 1)+K.int_shape(x)[1:])(x)
                      for x in difference]

        # Compute path steps.
        multiply_with_linspace = ilayers.MultiplyWithLinspace(
            0, 1,
            n=self._augment_by_n,
            axis=1)
        path_steps = [multiply_with_linspace(d) for d in difference]

        reference_inputs = self._keras_get_constant_inputs()
        ret = [keras.layers.Add()([x, p]) for x, p in zip(reference_inputs, path_steps)]
        ret = [ilayers.Reshape((-1,)+K.int_shape(x)[2:])(x) for x in ret]
        return ret

    def _reduce(self, X):
        tmp = super(PathIntegrator, self)._reduce(X)
        difference = self._keras_difference
        del self._keras_difference

        return [keras.layers.Multiply()([x, d])
                for x, d in zip(tmp, difference)]

    def _get_state(self):
        state = super(PathIntegrator, self)._get_state()
        state.update({"reference_inputs": self._reference_inputs})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        reference_inputs = state.pop("reference_inputs")
        kwargs = super(PathIntegrator, clazz)._state_to_kwargs(state)
        kwargs.update({"reference_inputs": reference_inputs})
        # We use steps instead.
        kwargs.update({"steps": kwargs["augment_by_n"]})
        del kwargs["augment_by_n"]
        return kwargs
