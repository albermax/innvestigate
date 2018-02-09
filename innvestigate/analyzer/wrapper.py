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
]


###############################################################################
###############################################################################
###############################################################################


class WrapperBase(base.AnalyzerBase):

    properties = {
        "name": "WrapperBase",
        "show_as": "rgb",
    }

    def __init__(self, subanalyzer, *args, **kwargs):
        self._subanalyzer = subanalyzer
        model = None
        return super(WrapperBase, self).__init__(model,
                                                 *args, **kwargs)

    def analyze(self, *args, **kwargs):
        return self._subanalyzer.analyze(*args, **kwargs)

    def _get_state(self):
        state = super(WrapperBase, self)._get_state()
        class_name, state = self._subanalyzer.save()
        state.update({"subanalyzer_class_name": class_name})
        state.update({"subanalyzer_state": state})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        class_name = state.pop("subanalyzer_class_name")
        state = state.pop("subanalyzer_state")
        kwargs = super(WrapperBase, clazz)._state_to_kwargs(state)
        subanalyzer = base.AnalyzerBase.load(class_name, state)
        kwargs.update({"subanalyzer": subanalyzer})
        return kwargs


###############################################################################
###############################################################################
###############################################################################


class AugmentReduceBase(WrapperBase):

    properties = {
        "name": "AugmentReduceBase",
        "show_as": "rgb",
    }

    def __init__(self, subanalyzer, *args, augment_by_n=2, **kwargs):
        self._augment_by_n = augment_by_n
        ret = super(AugmentReduceBase, self).__init__(subanalyzer,
                                                      *args, **kwargs)

        self._keras_based_augment_reduce = False
        if isinstance(self._subanalyzer, base.AnalyzerNetworkBase):
            # Take the keras analyzer model and
            # add augment and reduce functionality.
            self._keras_based_augment_reduce = True

            if self._subanalyzer._n_debug_output > 0:
                raise Exception("No debug output at subanalyzer is supported.")

            model = self._subanalyzer._analyzer_model
            inputs = model.inputs[:self._subanalyzer._n_data_input]
            extra_inputs = model.inputs[self._subanalyzer._n_data_input:]
            outputs = model.outputs[:self._subanalyzer._n_data_input]
            extra_outputs = model.outputs[self._subanalyzer._n_data_input:]

            if len(extra_outputs) > 0:
                raise Exception("No extra output is allowed "
                                "with this wrapper.")

            new_inputs = iutils.listify(self._keras_based_augment(inputs))
            tmp = iutils.listify(kutils.easy_apply(model, new_inputs))
            new_outputs = iutils.listify(self._keras_based_reduce(tmp))

            new_model = keras.models.Model(
                inputs=inputs+extra_inputs,
                outputs=new_outputs+extra_outputs)
            new_model.compile(optimizer="sgd", loss="mse")
            self._subanalyzer._analyzer_model = new_model

        return ret

    def analyze(self, X, *args, **kwargs):
        if self._keras_based_augment_reduce is True:
            return self._subanalyzer.analyze(X, *args, **kwargs)
        else:
            return_list = isinstance(X, list)

            X = self._python_based_augment(iutils.listify(X))
            ret = self._subanalyzer.analyze(X, *args, **kwargs)
            ret = self._python_based_reduce(ret)

            if return_list is True:
                return ret
            else:
                return ret[0]

    def _python_based_augment(self, X):
        return [np.repeat(x, self._augment_by_n, axis=0) for x in X]

    def _python_based_reduce(self, X):
        tmp = [x.reshape((-1, self._augment_by_n)+x.shape[1:]) for x in X]
        tmp = [x.mean(axis=1) for x in tmp]
        return tmp

    def _keras_based_augment(self, X):
        repeat = ilayers.Repeat(self._augment_by_n, axis=0)
        return [repeat(x) for x in iutils.listify(X)]

    def _keras_based_reduce(self, X):
        X_shape = [K.int_shape(x) for x in iutils.listify(X)]
        reshape = [ilayers.Reshape((-1, self._augment_by_n)+shape[1:])
                   for shape in X_shape]
        mean = ilayers.Mean(axis=0)

        return [mean(reshape_x(x)) for x, reshape_x in zip(X, reshape)]

    def _get_state(self):
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


class GaussianSmoother(WrapperBase):

    properties = {
        "name": "GaussianSmoother",
        "show_as": "rgb",
    }

    def __init__(self, subanalyzer, *args, noise_scale=1, **kwargs):
        self._noise_scale = noise_scale
        return super(GaussianSmoother, self).__init__(subanalyzer,
                                                      *args, **kwargs)

    def _python_based_augment(self, X):
        tmp = super(GaussianSmoother, self)._python_based_augment(X)
        return [x + np.random.normal(0, self._noise_scale, size=x.shape)
                for x in X]

    def _keras_based_augment(self, X):
        tmp = super(GaussianSmoother, self)._keras_based_augment(X)
        noise = ilayers.AddGaussianNoise(mean=0, scale=self._noise_scale)
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
