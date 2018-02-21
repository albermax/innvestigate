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
import keras.optimizers
import keras.utils
import numpy as np


from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import graph as kgraph


__all__ = [
    "BasePattern",
    "PatternComputer",
]


###############################################################################
###############################################################################
###############################################################################


class BasePattern(object):

    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def has_pattern(self):
        return kgraph.contains_kernel(self.layer)

    def stats_from_batch(self):
        raise NotImplementedError()

    def compute_pattern(self):
        raise NotImplementedError()


class DummyPattern(BasePattern):

    def get_stats_from_batch(self):
        # code is not ready for shared layers
        assert kgraph.get_layer_inbound_count(self.layer) == 1
        Xs, Ys = kgraph.get_layer_neuronwise_io(self.layer)
        self.mean_x = ilayers.RunningMeans()

        count = ilayers.CountNonZero(axis=0)(Ys[0])
        sum_x = ilayers.Dot()([ilayers.Transpose()(Xs[0]), Ys[0]])

        mean_x, count_x = self.mean_x([sum_x, count])

        # Return dummy output to have connected graph!
        return ilayers.Sum(axis=None)(count_x)

    def compute_pattern(self):
        return self.mean_x.get_weights()[0]


class LinearPattern(BasePattern):

    def _get_neuron_mask(self):
        layer = kgraph.get_layer_wo_activation(self.layer, keep_bias=True)
        Y = kgraph.get_layer_neuronwise_io(layer,
                                           return_i=False, return_o=True)
        return ilayers.OnesLike()(Y)

    def get_stats_from_batch(self):
        layer = kgraph.get_layer_wo_activation(self.layer, keep_bias=False)
        X, Y = kgraph.get_layer_neuronwise_io(layer)
        X, Y = X[0], Y[0]

        self.mean_x = ilayers.RunningMeans()
        self.mean_y = ilayers.RunningMeans()
        self.mean_xy = ilayers.RunningMeans()

        # Compute mask and active neuron counts.
        mask = ilayers.AsFloatX()(self._get_neuron_mask())
        Y_masked = keras.layers.multiply([Y, mask])
        count = ilayers.CountNonZero(axis=0)(mask)
        count_all = ilayers.Sum(axis=0)(ilayers.OnesLike()(mask))

        # Get means ...
        def norm(x, count):
            return ilayers.SafeDivide(factor=1)([x, count])

        # ... along active neurons.
        mean_x = norm(ilayers.Dot()([ilayers.Transpose()(X), mask]), count)
        mean_xy = norm(ilayers.Dot()([ilayers.Transpose()(X), Y_masked]),
                       count)

        _, a = self.mean_x([mean_x, count])
        _, b = self.mean_xy([mean_xy, count])

        # ... along all neurons.
        mean_y = norm(ilayers.Sum(axis=0)(Y_masked), count_all)
        _, c = self.mean_y([mean_y, count_all])

        # Create a dummy output to have a connected graph.
        # Needs to have the shape (mb_size, 1)
        dummy = keras.layers.Average()([a, b, c])
        return ilayers.Sum(axis=None)(dummy)

    def compute_pattern(self):

        def safe_divide(a, b):
            return a / (b + (b == 0))

        W = kgraph.get_kernel(self.layer)
        W2D = W.reshape((-1, W.shape[-1]))

        mean_x = self.mean_x.get_weights()[0]
        mean_y = self.mean_y.get_weights()[0]
        mean_xy = self.mean_xy.get_weights()[0]

        ExEy = mean_x * mean_y
        EyEy = mean_y * mean_y
        cov_xy = mean_xy - ExEy

        w_cov_xy = np.diag(np.dot(W2D.T, cov_xy))
        A = safe_divide(cov_xy, w_cov_xy)

        # update length
        if True:
            norm = np.diag(np.dot(W2D.T, A))
            A = safe_divide(A, norm)

        # check pattern
        if False:
            tmp = np.diag(np.dot(W2D.T, A))
            print("pattern_check", W.shape, tmp.min(), tmp.max())

        return A.reshape(W.shape)


class ReluPositivePattern(LinearPattern):

    def _get_neuron_mask(self):
        layer = kgraph.get_layer_wo_activation(self.layer, keep_bias=True)
        Y = kgraph.get_layer_neuronwise_io(layer,
                                           return_i=False, return_o=True)
        return ilayers.GreaterThanZero()(Y[0])


class ReluNegativePattern(LinearPattern):

    def _get_neuron_mask(self):
        layer = kgraph.get_layer_wo_activation(self.layer, keep_bias=True)
        Y = kgraph.get_layer_neuronwise_io(layer,
                                           return_i=False, return_o=True)
        return ilayers.LessThanZero()(Y[0])


def get_pattern_class(pattern_type):
    return {
        "dummy": DummyPattern,

        "linear": LinearPattern,
        "relu": ReluPositivePattern,
        "relu.positive": ReluPositivePattern,
        "relu.negative": ReluNegativePattern,
    }.get(pattern_type, pattern_type)


###############################################################################
###############################################################################
###############################################################################


class PatternComputer(object):

    def __init__(self, model,
                 pattern_type="linear",
                 # todo: this options seems to be buggy,
                 # if it sequential tensorflow still pushes all models to gpus
                 compute_layers_in_parallel=True,
                 gpus=None):
        self.model = model
        pattern_types = iutils.listify(pattern_type)
        self.pattern_types = {k: get_pattern_class(k)
                              for k in pattern_types}
        self.compute_layers_in_parallel = compute_layers_in_parallel
        self.gpus = gpus

        # create pattern instances and collect keras outputs
        self._work_sequence = []
        self._pattern_instances = {k: [] for k in self.pattern_types}
        computer_outputs = []
        # Broadcaster has shape (mb, 1)
        # Todod: does not work for tensors
        reduce_axes = list(range(len(K.int_shape(model.inputs[0]))))[1:]
        dummy_broadcaster = ilayers.Sum(axis=reduce_axes,
                                        keepdims=True)(model.inputs[0])

        def broadcast(x):
            return ilayers.Broadcast()([dummy_broadcaster, x])

        # todo: this does not work with more nodes or containers!
        for layer_id, layer in enumerate(model.layers):
            if kgraph.is_container(layer):
                raise Exception("Container in container is not suppored!")
            for pattern_type, clazz in six.iteritems(self.pattern_types):
                pinstance = clazz(model, layer)
                if pinstance.has_pattern() is False:
                    continue
                self._pattern_instances[pattern_type].append(pinstance)
                dummy_output = pinstance.get_stats_from_batch()
                # Broadcast dummy_output to right shape.
                computer_outputs += iutils.listify(broadcast(dummy_output))

        # initialize the keras outputs
        self._n_computer_outputs = len(computer_outputs)
        if self.compute_layers_in_parallel is True:
            self._computers = [
                keras.models.Model(inputs=model.inputs,
                                   outputs=computer_outputs)
            ]
        else:
            self._computers = [
                keras.models.Model(inputs=model.inputs,
                                   outputs=computer_output)
                for computer_output in computer_outputs
            ]

        # distribute computation on more gpus
        if self.gpus is not None and self.gpus > 1:
            raise NotImplementedError("Not supported yet.")
            self._computers = [keras.utils.multi_gpu_model(tmp, gpus=self.gpus)
                               for tmp in self._computers]
        # todo: model compiling?
        pass

    def compute(self, X, batch_size=32, verbose=0):
        generator = iutils.BatchSequence(X, batch_size)
        return self.compute_generator(generator, verbose=verbose)

    def compute_generator(self, generator, **kwargs):
        if not hasattr(self, "_computers"):
            raise Exception("One shot computer. Already used.")

        # We don't do gradient updates.
        class NoOptimizer(keras.optimizers.Optimizer):
            def get_updates(self, *args, **kwargs):
                return []
        optimizer = NoOptimizer()
        # We only go over the training data once.
        if "epochs" in kwargs and kwargs["epochs"] != 1:
            raise ValueError("Pattern are computed with "
                             "a closed form solution. "
                             "Only need to do one epoch.")
        kwargs["epochs"] = 1

        if self.compute_layers_in_parallel is True:
            n_dummy_outputs = self._n_computer_outputs
        else:
            n_dummy_outputs = 1

        # Augment the input with dummy targets.
        def get_dummy_targets(Xs):
            n, dtype = Xs[0].shape[0], Xs[0].dtype
            dummy = np.ones(shape=(n, 1), dtype=dtype)
            return [dummy for _ in range(n_dummy_outputs)]

        if isinstance(generator, keras.utils.Sequence):
            generator = iutils.TargetAugmentedSequence(generator,
                                                       get_dummy_targets)
        else:
            base_generator = generator

            def generator(*args, **kwargs):
                for Xs in base_generator(*args, **kwargs):
                    Xs = iutils.listify(Xs)
                    yield Xs, get_dummy_targets(Xs)

        # Compile models.
        for computer in self._computers:
            computer.compile(optimizer=optimizer, loss=lambda x, y: x)

        # Compute pattern statistics.
        for computer in self._computers:

            computer.fit_generator(generator, **kwargs)

        # retrieve the actual patterns
        pis = self._pattern_instances
        patterns = {ptype: [tmp.compute_pattern() for tmp in pis[ptype]]
                    for ptype in self.pattern_types}

        # free memory
        del self._computers
        del self._work_sequence
        del self._pattern_instances

        if len(self.pattern_types) == 1:
            return patterns[list(self.pattern_types.keys())[0]]
        else:
            return patterns
