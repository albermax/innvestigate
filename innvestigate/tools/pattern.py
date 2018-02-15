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
        self._stats = None

    def _stats_to_dict(self, stats):
        return {k: stats[idx]
                for idx, k in enumerate(self._stats_keys)}

    def _stats_to_list(self, stats):
        return [stats[k] for k in self._stats_keys]

    def has_pattern(self):
        return kgraph.contains_kernel(self.layer)

    def stats_from_batch(self):
        raise NotImplementedError()

    def _update_stats(self, stats, batch_stats):
        raise NotImplementedError()

    def update_stats(self, batch_stats):
        batch_stats = self._stats_to_dict(batch_stats)
        if self._stats is None:
            self._stats = batch_stats
        else:
            # force in place updates
            self._update_stats(self._stats, batch_stats)
        pass

    def compute_pattern(self):
        raise NotImplementedError()


class DummyPattern(BasePattern):

    _stats_keys = ["sum"]

    def get_stats_from_batch(self):
        # code is not ready for shared layers
        assert kgraph.get_layer_inbound_count(self.layer) == 1
        Xs, Ys = kgraph.get_layer_neuronwise_io(self.layer)
        return ilayers.Sum()(Xs)

    def _update_stats(self, stats, batch_stats):
        stats["sum"] += batch_stats["sum"]
        pass

    def compute_pattern(self):
        return self._stats_to_list(self._stats)


class LinearPattern(BasePattern):

    _stats_keys = [
        "count",
        "mean_x",
        "mean_y",
        "mean_xy",
        "mean_yy",
    ]

    def _get_neuron_mask(self):
        layer = kgraph.get_layer_wo_activation(self.layer, keep_bias=True)
        Y = kgraph.get_layer_neuronwise_io(layer,
                                           return_i=False, return_o=True)
        return ilayers.OnesLike()(Y)

    def get_stats_from_batch(self):
        layer = kgraph.get_layer_wo_activation(self.layer, keep_bias=False)
        X, Y = kgraph.get_layer_neuronwise_io(layer)
        X, Y = X[0], Y[0]

        mask = ilayers.AsFloatX()(self._get_neuron_mask())
        Y_masked = keras.layers.multiply([Y, mask])
        count = ilayers.CountNonZero(axis=0)(mask)

        def norm(x):
            return ilayers.SafeDivide(factor=1)([x, count])

        mean_x = norm(ilayers.Dot()([ilayers.Transpose()(X), mask]))
        mean_y = norm(ilayers.Sum(axis=0)(Y_masked))
        mean_xy = norm(ilayers.Dot()([ilayers.Transpose()(X), Y_masked]))
        mean_yy = norm(ilayers.Sum(axis=0)(ilayers.Square()(Y_masked)))
        return [count, mean_x, mean_y, mean_xy, mean_yy]

    def _update_stats(self, stats, batch_stats):
        """
        Updating stats neuronwise with a running average.
        """
        old_count, new_count = stats["count"], batch_stats["count"]
        total_count = old_count+new_count

        def safe_divide(a, b):
            return K.cast_to_floatx(a) / np.maximum(b, 1)
        factor_old = safe_divide(old_count, total_count)
        factor_new = safe_divide(new_count, total_count)

        def update(old, new):
            # old_count/total_count * old_val + new_count/total_count * new_val
            old *= factor_old
            old += factor_new * new
            pass

        stats["count"] = total_count
        for k in self._stats_keys[1:]:
            update(stats[k], batch_stats[k])
        pass

    def compute_pattern(self):
        W = kgraph.get_kernel(self.layer)

        def safe_divide(a, b):
            return a / (b + (b == 0))

        ExEy = self._stats["mean_x"] * self._stats["mean_y"]
        EyEy = self._stats["mean_y"] * self._stats["mean_y"]
        cov_xy = self._stats["mean_xy"] - ExEy
        sq_sigma_y = self._stats["mean_yy"] - EyEy

        A = safe_divide(cov_xy, sq_sigma_y)

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
        # todo: this does not work with more nodes or containers!
        for layer_id, layer in enumerate(model.layers):
            if kgraph.is_container(layer):
                raise Exception("Container in container is not suppored!")
            for pattern_type, clazz in six.iteritems(self.pattern_types):
                pinstance = clazz(model, layer)
                if pinstance.has_pattern() is False:
                    continue
                self._pattern_instances[pattern_type].append(pinstance)
                batch_stats = iutils.listify(pinstance.get_stats_from_batch())

                n = len(computer_outputs)
                self._work_sequence.append((n, n+len(batch_stats), pinstance))
                computer_outputs += iutils.listify(batch_stats)

        # initialize the keras outputs
        if self.compute_layers_in_parallel is True:
            self._computers = [
                keras.models.Model(inputs=model.inputs,
                                   outputs=computer_outputs)
            ]
        else:
            self._computers = [
                keras.models.Model(inputs=model.inputs,
                                   outputs=computer_outputs[i:j])
                for i, j, _ in self._work_sequence
            ]

        # distribute computation on more gpus
        if self.gpus is not None and self.gpus > 1:
            self._computers = [keras.utils.multi_gpu_model(tmp, gpus=self.gpus)
                               for tmp in self._computers]
        # todo: model compiling?
        pass

    def compute(self, X, batch_size=32, verbose=0):
        generator = iutils.BatchSequence(X, batch_size)
        return self.compute_generator(generator, verbose=verbose)

    def compute_generator(self,
                          generator,
                          steps=None,
                          max_queue_size=10,
                          workers=1,
                          use_multiprocessing=False,
                          verbose=0):
        if not hasattr(self, "_computers"):
            raise Exception("One shot computer. Already used.")

        if steps is None:
            steps = len(generator)

        # the next part of the code is copied and modified from
        # from keras model.predict_generator
        steps_done = 0
        wait_time = 0.01
        is_sequence = isinstance(generator, keras.utils.Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
        enqueuer = None

        try:
            if is_sequence:
                enqueuer = keras.utils.OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = keras.utils.GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

            if verbose == 1:
                progbar = keras.utils.Progbar(target=steps)

            while steps_done < steps:
                generator_output = next(output_generator)
                if isinstance(generator_output, tuple):
                    # Compatibility with the generators
                    # used for training.
                    if len(generator_output) == 2:
                        x, _ = generator_output
                    elif len(generator_output) == 3:
                        x, _, _ = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                else:
                    # Assumes a generator that only
                    # yields inputs (not targets and sample weights).
                    x = generator_output

                batch_state = []
                for computer in self._computers:
                    batch_state += iutils.listify(computer.predict_on_batch(x))

                for i, j, pinstance in self._work_sequence:
                    pinstance.update_stats(batch_state[i:j])

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            if enqueuer is not None:
                enqueuer.stop()

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
