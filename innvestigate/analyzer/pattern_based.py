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
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph


__all__ = [
    "PatternNet",
    "PatternAttribution",
]


###############################################################################
###############################################################################
###############################################################################


class PatternNet(base.OneEpochTrainerMixin, base.ReverseAnalyzerBase):
    """PatternNet analyzer.

    Applies the "PatternNet" algorithm to analyze the model.

    :param model: A Keras model.
    :param patterns: Pattern computed by
      :class:`innvestigate.tools.PatternComputer`. If None :func:`fit` needs
      to be called.
    :param allow_lambda_layers: Approximate lambda layers with the gradient.
    """

    def __init__(self,
                 model,
                 patterns=None,
                 allow_lambda_layers=False,
                 **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check":
                lambda layer: kchecks.contains_activation(
                    layer,
                    activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            {
                "check": lambda layer: not kchecks.only_relu_activation(layer),
                "type": "warning",
                "message": ("PatternNet is not well defined for "
                            "networks with non-ReLU activations."),
            },
            {
                "check":
                lambda layer: not kchecks.is_convnet_layer(layer),
                "type": "warning",
                "message": ("PatternNet is only well defined for "
                            "convolutional neural networks."),
            },
            {
                "check":
                lambda layer: kchecks.is_average_pooling(layer),
                "type": "warning",
                "message": ("PatternNet is only well defined for "
                            "max-pooling pooling layers."),
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

        self._patterns = patterns
        if self._patterns is not None:
            # copy pattern references
            self._patterns = list(patterns)
        self._allow_lambda_layers = allow_lambda_layers

        # Pattern projections can lead to +-inf value with long networks.
        # We are only interested in the direction, therefore it is save to
        # Prevent this by projecting the values in bottleneck layers to +-1.
        if not kwargs.get("reverse_project_bottleneck_layers", True):
            warnings.warn("The standard setting for "
                          "'reverse_project_bottleneck_layers' "
                          "is overwritten.")
        else:
            kwargs["reverse_project_bottleneck_layers"] = True

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

                return grad_pattern(Xs+pattern_Ys+tmp)

        self._conditional_mappings = [
            (kchecks.contains_kernel, ReverseLayer),
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
        state.update({"allow_lambda_layers": self._allow_lambda_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        patterns = state.pop("patterns")
        allow_lambda_layers = state.pop("allow_lambda_layers")
        kwargs = super(PatternNet, clazz)._state_to_kwargs(state)
        kwargs.update({"patterns": patterns,
                       "allow_lambda_layers": allow_lambda_layers})
        return kwargs


class PatternAttribution(PatternNet):
    """PatternAttribution analyzer.

    Applies the "PatternAttribution" algorithm to analyze the model.

    :param model: A Keras model.
    :param patterns: Pattern computed by
      :class:`innvestigate.tools.PatternComputer`. If None :func:`fit` needs
      to be called.
    :param allow_lambda_layers: Approximate lambda layers with the gradient.
    """

    def _prepare_pattern(self, layer, state, pattern):
        weights = layer.get_weights()
        tmp = [pattern.shape == x.shape for x in weights]
        if np.sum(tmp) != 1:
            raise Exception("Cannot match pattern to kernel.")
        weight = weights[np.argmax(tmp)]
        return np.multiply(pattern, weight)
