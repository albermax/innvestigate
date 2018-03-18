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

import keras.activations
import keras.backend as K
import keras.models
import keras
import numpy as np
import warnings


from . import base
from .. import layers as ilayers
from .. import utils
from .. import tools as itools
from ..utils import keras as kutils
from ..utils.keras import graph as kgraph


__all__ = [
    "PatternNet",
    "PatternAttribution",
]


###############################################################################
###############################################################################
###############################################################################


class PatternNet(base.OneEpochTrainerMixin, base.ReverseAnalyzerBase):

    def __init__(self, model, patterns=None, **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kgraph.contains_activation(
                    layer, activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            # todo: be more specific here:
            {
                "check": lambda layer: not kgraph.is_relu_convnet_layer(layer),
                "type": "warning",
                "message": ("PatternNet is only well defined for "
                            "convolutional neural networks with "
                            "relu activations."),
            },
        ]

        self._patterns = patterns
        if self._patterns is not None:
            # copy pattern references
            self._patterns = list(patterns)

        # Pattern projections can lead to +-inf value with long networks.
        # We are only interested in the direction, therefore it is save to
        # Prevent this by projecting the values in bottleneck layers to +-1.
        if "reverse_project_bottleneck_layers" in kwargs:
            warnings.warn("The standard setting for "
                          "'reverse_project_bottleneck_layers' "
                          "is overwritten.")
        else:
            kwargs["reverse_project_bottleneck_layers"] = False

        return super(PatternNet, self).__init__(model, **kwargs)

    def _prepare_pattern(self, layer, state, pattern):
        return pattern

    def _create_analysis(self, *args, **kwargs):
        # shared information among the created reverse mappings
        # todo: make this more flexible.
        tmp_pattern_idx_stack = list(range(len(self._patterns)))

        def get_pattern(layer, state):
            if self._patterns is None:
                raise Exception("Patterns are required. "
                                "Either train them or "
                                "pass them to the constructor.")

            pattern = self._patterns[tmp_pattern_idx_stack.pop(0)]
            return self._prepare_pattern(layer, state, pattern)

        class ReverseLayer(kgraph.ReverseMappingBase):

            def __init__(self, layer, state):
                # we want to apply the filter weights on the forward pass
                # on the backward pass we want copy the gradient computation
                # except that using the pattern weights instead of the filter w
                # i.e. we need to revert the activation with the forward
                # activations to keep the relu patterns of the gradient comp.
                # this is the reason for splitting the gradient mimicking:
                config = layer.get_config()

                activation = None
                if "activation" in config:
                    activation = config["activation"]
                    config["activation"] = None
                self._act_layer = keras.layers.Activation(
                    activation,
                    name="reversed_act_%s" % config["name"])

                self._kernel_layer = kgraph.copy_layer_wo_activation(
                    layer, name_template="reversed_kernel_%s")

                # replace kernel weights with pattern weights
                pattern_weights = layer.get_weights()
                pattern = get_pattern(layer, state)
                # assume that only one matches
                tmp = [pattern.shape == x.shape for x in pattern_weights]
                if np.sum(tmp) != 1:
                    raise Exception("Cannot match pattern to kernel.")
                pattern_weights[np.argmax(tmp)] = pattern
                self._pattern_layer = kgraph.copy_layer_wo_activation(
                    layer,
                    name_template="reversed_pattern_%s",
                    weights=pattern_weights)

            def apply(self, Xs, Ys, reversed_Ys, reverse_state):
                act_Xs = kutils.apply(self._kernel_layer, Xs)
                act_Ys = kutils.apply(self._act_layer, act_Xs)
                pattern_Ys = kutils.apply(self._pattern_layer, Xs)

                grad_act = ilayers.GradientWRT(len(act_Xs))
                grad_pattern = ilayers.GradientWRT(len(Xs))

                linear_activations = [None, keras.activations.get("linear")]
                if self._act_layer.activation in linear_activations:
                    tmp = reversed_Ys
                else:
                    # if linear activation this behaves strange
                    tmp = utils.to_list(grad_act(act_Xs+act_Ys+reversed_Ys))

                ret = grad_pattern(Xs+pattern_Ys+tmp)
                ret = ilayers.Project()(ret)
                return ret

        self._conditional_mappings = [
            (kgraph.contains_kernel, ReverseLayer),
        ]

        ret = super(PatternNet, self)._create_analysis(*args, **kwargs)
        if len(tmp_pattern_idx_stack) != 0:
            raise Exception("Not all patterns consumed. Something is wrong.")

        return ret

    def _fit_generator(self,
                       generator,
                       steps_per_epoch=None,
                       epochs=1,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0,
                       disable_no_training_warning=None,
                       pattern_type="linear",
                       **kwargs):

        if isinstance(pattern_type, (list, tuple)):
            raise ValueError("Only one pattern type allowed. "
                             "Please pass string.")

        computer = itools.PatternComputer(self._model,
                                          pattern_type=pattern_type,
                                          **kwargs)

        self._patterns = computer.compute_generator(
            generator,
            steps_per_epoch=steps_per_epoch,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=verbose)
        pass

    def _get_state(self):
        state = super(PatternNet, self)._get_state()
        state.update({"patterns": self._patterns})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        patterns = state.pop("patterns")
        kwargs = super(PatternNet, clazz)._state_to_kwargs(state)
        kwargs.update({"patterns": patterns})
        return kwargs


class PatternAttribution(PatternNet):

    def _prepare_pattern(self, layer, state, pattern):
        weights = layer.get_weights()
        tmp = [pattern.shape == x.shape for x in weights]
        if np.sum(tmp) != 1:
            raise Exception("Cannot match pattern to kernel.")
        weight = weights[np.argmax(tmp)]
        return np.multiply(pattern, weight)
