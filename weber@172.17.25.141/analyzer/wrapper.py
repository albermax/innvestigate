# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import zip


###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models
import tensorflow.keras.backend as K
import numpy as np

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export

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

class ConstantInputLayer(base_layer.Layer):
  """
  tf.keras Input Layer, for constant inputs
  """

  def __init__(self,
               input_tensor=None,
               sparse=False,
               name=None,
               ragged=False,
               **kwargs):

    if kwargs:
      raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(backend.get_uid(prefix))

    dtype = backend.dtype(input_tensor)

    super(ConstantInputLayer, self).__init__(dtype=dtype, name=name)
    self.built = True
    self.sparse = sparse
    self.ragged = ragged
    self.batch_size = None
    self.supports_masking = True
    self.is_placeholder = False
    self._batch_input_shape = tuple(input_tensor.shape.as_list())

    # Create an input node to add to self.outbound_node
    # and set output_tensors' _keras_history.
    input_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
    input_tensor._keras_mask = None
    node_module.Node(
        self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=[input_tensor],
        output_tensors=[input_tensor])

  def get_config(self):
    config = {
        'batch_input_shape': self._batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'ragged': self.ragged,
        'name': self.name
    }
    return config

  @property
  def _trackable_saved_model_saver(self):
    return layer_serialization.InputLayerSavedModelSaver(self)
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

        super(AugmentReduceBase, self).__init__(subanalyzer,
                                                *args, **kwargs)

        if isinstance(self._subanalyzer, base.AnalyzerNetworkBase):
            # Take the keras analyzer model and
            # add augment and reduce functionality.
            self._keras_based_augment_reduce = True
        else:
            raise NotImplementedError("Keras-based subanalyzer required.")

    def analyze(self, X, *args, **kwargs):
        if self._keras_based_augment_reduce is True:

            if not hasattr(self._subanalyzer, "_analyzer_model"):
                self._subanalyzer.create_analyzer_model()

            augmented = self._augment(X)
            analyzed = {}
            for X in augmented:
                hm = self._subanalyzer.analyze(X, *args, **kwargs)
                for key in hm.keys():
                    if key not in analyzed.keys():
                        analyzed[key] = []
                    analyzed[key].append(hm[key])
            ret = self._reduce(analyzed)

            return ret
        else:
            raise DeprecationWarning("Not supported anymore.")

    def _augment(self, X):
        #creates augment_by_n samples for each original sample in X

        # X is array-like
        repeat = [X for _ in range(self._augment_by_n)]

        return repeat

    def _reduce(self, X):
        #reduces the augmented samples to original number of samples in X

        # X is a dict for each layer that is explained

        means = {}
        for key in X.keys():
            means[key] = np.mean(X[key], axis=0)

        return means


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
        X = super(GaussianSmoother, self)._augment(X)
        ins, rev = self._subanalyzer._analyzer_model._reverse_model
        if len(ins) == 1:
            for i, x in enumerate(X):
                noise = np.random.normal(0, self._noise_scale, np.shape(x))
                X[i] += noise
        else:
            for i, x_ins in enumerate(X):
                for j, x in enumerate(x_ins):
                    noise = np.random.normal(0, self._noise_scale, np.shape(x))
                    X[i][j] += noise
        return X


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
        super(PathIntegrator, self).__init__(subanalyzer,
                                             *args,
                                             augment_by_n=steps,
                                             **kwargs)

    def analyze(self, X, *args, **kwargs):
        explained_layer_names = kwargs.pop("explained_layer_names", None)
        if explained_layer_names is not None and len(explained_layer_names) > 0:
            raise ValueError("Intermediate explanations are not available for Integrated Gradients")

        return super(PathIntegrator, self).analyze(X, *args, **kwargs)

    def _augment(self, X):
        X = super(PathIntegrator, self)._augment(X)

        ins, rev = self._subanalyzer._analyzer_model._reverse_model
        self.difference = {}
        if len(ins) == 1:
            ret = []
            for i, x in enumerate(X):
                difference = (np.array(x) - self._reference_inputs)
                #X is only repeated _augment_by_n times by superclass method --> difference is the same each time
                self.difference[ins[0].name] = difference
                step_size = difference / (self._augment_by_n-1)
                ret.append(self._reference_inputs + step_size*i)
        else:
            ret = []
            for i, x_ins in enumerate(X):
                ret.append([])
                for j, x in enumerate(x_ins):
                    difference = (x - self._reference_inputs)
                    # X is only repeated _augment_by_n times by superclass method --> difference is the same each time
                    self.difference[ins[j].name] = difference
                    step_size = difference / (self._augment_by_n - 1)
                    ret[-1].append(self._reference_inputs + step_size * j)

        return ret

    def _reduce(self, X):
        X = super(PathIntegrator, self)._reduce(X)

        ret = {}
        for key in X.keys():
            ret[key] = self.difference[key] * X[key]

        return ret
