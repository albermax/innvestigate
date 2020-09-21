# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################

import tensorflow as tf
import tensorflow.keras.layers as keras_layers

from . import base
from . import reverse_map
from . import wrapper
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph

__all__ = [
    "Gradient",

    "InputTimesGradient",

    "Deconvnet",
    "GuidedBackprop",

    "IntegratedGradients",

    "SmoothGrad",
]


###############################################################################
###############################################################################
###############################################################################

class GradientOnesReplacementLayer(reverse_map.GradientReplacementLayer):
    """
    Simple extension of GradientHeadMapReplacementLayer
    * Explains by computing gradients of outputs w.r.t. inputs of layer
    * Headmapping equals one
    """

    def __init__(self, *args, **kwargs):
        super(GradientOnesReplacementLayer, self).__init__(*args, **kwargs, r_init_constant=1)

class Gradient(base.ReverseAnalyzerBase):
    """Gradient analyzer.

    Returns as analysis the function value with respect to the input.
    The gradient is computed via the library's network reverting.

    :param model: A Keras model.
    """

    def __init__(self, model, postprocess=None, **kwargs):

        if postprocess not in [None, "abs", "square"]:
            raise ValueError("Parameter 'postprocess' must be either "
                             "None, 'abs', or 'square'.")
        self._postprocess = postprocess

        self._add_model_softmax_check()

        super(Gradient, self).__init__(model, **kwargs)

    def _default_reverse_mapping(self, layer):
        return GradientOnesReplacementLayer

    def _postprocess_analysis(self, hm):
        hm = super(Gradient, self)._postprocess_analysis(hm)

        for key in hm.keys():
            if self._postprocess == "abs":
                hm[key] = ilayers.Abs()(hm[key]).numpy()
            elif self._postprocess == "square":
                hm[key] = ilayers.Square()(hm[key]).numpy()

        return hm


###############################################################################
###############################################################################
###############################################################################

class InputTimesGradientReplacementLayer(GradientOnesReplacementLayer):
    """
    ReplacementLayer for Input*Gradient
    """

    def __init__(self, *args, **kwargs):
        super(InputTimesGradientReplacementLayer, self).__init__(*args, **kwargs)

    def try_explain(self, reversed_outs):
        """
        self.explanation here is input*gradient, however only gradient is sent to callbacks
        """
        # aggregate explanations
        if reversed_outs is not None:
            if self.reversed_output_vals is None:
                self.reversed_output_vals = []
            self.reversed_output_vals.append(reversed_outs)

        # last layer or aggregation finished
        if self.reversed_output_vals is None or len(self.reversed_output_vals) == len(self.layer_next):
            # apply post hook: explain
            if self.hook_vals is None:
                raise ValueError(
                    "self.hook_vals should contain values at this point. Is self.wrap_hook working correctly?")
            input_vals = self.input_vals
            if len(input_vals) == 1:
                input_vals = input_vals[0]

            rev_outs = self.reversed_output_vals
            if rev_outs is not None:
                if len(rev_outs) == 1:
                    rev_outs = rev_outs[0]

            # print(self.name, np.shape(input_vals), np.shape(rev_outs), np.shape(self.hook_vals[0]))
            self.explanation = self.explain_hook(input_vals, rev_outs, self.hook_vals)

            #gradient*input specific
            explanation = self.explanation

            if len(self.input_shape) > 1:
                self.explanation = [e*i for e, i in zip(input_vals, self.explanation)]
            else:
                self.explanation = self.explanation * tf.cast(input_vals, self.explanation.dtype)

            # callbacks
            if self.callbacks is not None:
                # check if multiple inputs explained
                if len(self.callbacks) > 1 and not isinstance(explanation, list):
                    raise ValueError(self.name + ": This layer has " + str(
                        len(self.callbacks)) + " inputs, but no list of explanations was provided.")
                elif len(self.callbacks) > 1 and len(self.callbacks) != len(explanation):
                    raise ValueError(
                        self.name + ": This layer has " + str(len(self.callbacks)) + " inputs, but only " + str(
                            len(explanation)) + " explanations were computed")

                if len(self.callbacks) > 1:
                    for c, callback in enumerate(self.callbacks):
                        callback(explanation[c])
                else:
                    self.callbacks[0](explanation)

            # reset
            self.input_vals = None
            self.reversed_output_vals = None
            self.callbacks = None
            self.hook_vals = None

class InputTimesGradient(Gradient):
    """Input*Gradient analyzer.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):

        self._add_model_softmax_check()

        super(InputTimesGradient, self).__init__(model, **kwargs)

    def _default_reverse_mapping(self, layer):
        return InputTimesGradientReplacementLayer

###############################################################################
###############################################################################
###############################################################################

class DeconvnetReplacementLayer(reverse_map.ReplacementLayer):

    def __init__(self, layer, *args, **kwargs):
        self._activation = keras_layers.Activation("relu")
        self._layer_wo_relu = kgraph.copy_layer_wo_activation(
            layer,
            name_template="reversed_%s",
        )
        super(DeconvnetReplacementLayer, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)
            Ys = self._layer_wo_relu(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                outs = self._neuron_sel_and_head_map(outs, neuron_selection, r_init)
                Ys = self._neuron_sel_and_head_map(Ys, neuron_selection, r_init)

        return outs, Ys, tape

    def explain_hook(self, ins, reversed_outs, args):

        outs, Ys, tape = args

        # last layer
        if reversed_outs is None:
            reversed_outs = Ys

        # Apply relus conditioned on backpropagated values.
        Ys_wo_relu = self._layer_wo_relu(ins)

        if len(self.layer_next) > 1:
            reversed_outs = [self._activation(r) for r in reversed_outs]
            # Apply gradient.
            if len(self.input_shape) > 1:
                ret = [keras_layers.Add()([tape.gradient(Ys_wo_relu, i, output_gradients=r) for r in reversed_outs]) for i in ins]
            else:
                ret = keras_layers.Add()([tape.gradient(Ys_wo_relu, ins, output_gradients=r)  for r in reversed_outs])
        else:
            reversed_outs = self._activation(reversed_outs)
            # Apply gradient.
            if len(self.input_shape) > 1:
                ret = [tape.gradient(outs, i, output_gradients=reversed_outs) for i in ins]
            else:
                ret = tape.gradient(Ys_wo_relu, ins, output_gradients=reversed_outs)
        return ret

class Deconvnet(base.ReverseAnalyzerBase):
    """Deconvnet analyzer.

    Applies the "deconvnet" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "Deconvnet is only specified for networks with ReLU activations.",
            check_type="exception",
        )

        super(Deconvnet, self).__init__(model, **kwargs)

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: kchecks.contains_activation(layer, "relu"),
            DeconvnetReplacementLayer,
            name="deconvnet_reverse_relu_layer",
        )

        return super(Deconvnet, self)._create_analysis(*args, **kwargs)

class GuidedBackpropReplacementLayer(reverse_map.GradientReplacementLayer):

    def __init__(self, layer, *args, **kwargs):
        self._activation = keras_layers.Activation("relu")
        super(GuidedBackpropReplacementLayer, self).__init__(layer, *args, **kwargs)

    def explain_hook(self, ins, reversed_outs, args):

        outs, tape = args

        # last layer
        if reversed_outs is None:
            reversed_outs = outs

        if len(self.layer_next) > 1:
            reversed_outs = [self._activation(r) for r in reversed_outs]
            # Apply gradient.
            if len(self.input_shape) > 1:
                ret = [keras_layers.Add()([tape.gradient(outs, i, output_gradients=r) for r in reversed_outs]) for i in ins]
            else:
                ret = keras_layers.Add()([tape.gradient(outs, ins, output_gradients=r)  for r in reversed_outs])
        else:
            reversed_outs = self._activation(reversed_outs)
            # Apply gradient.
            if len(self.input_shape) > 1:
                ret = [tape.gradient(outs, i, output_gradients=reversed_outs) for i in ins]
            else:
                ret = tape.gradient(outs, ins, output_gradients=reversed_outs)
        return ret

class GuidedBackprop(base.ReverseAnalyzerBase):
    """Guided backprop analyzer.

    Applies the "guided backprop" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "GuidedBackprop is only specified for "
            "networks with ReLU activations.",
            check_type="exception",
        )

        super(GuidedBackprop, self).__init__(model, **kwargs)

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: kchecks.contains_activation(layer, "relu"),
            GuidedBackpropReplacementLayer,
            name="guided_backprop_reverse_relu_layer",
        )

        return super(GuidedBackprop, self)._create_analysis(*args, **kwargs)


###############################################################################
###############################################################################
###############################################################################

class IntegratedGradients(wrapper.PathIntegrator):
    """Integrated gradient analyzer.

    Applies the "integrated gradient" algorithm to analyze the model.

    :param model: A Keras model.
    :param steps: Number of steps to use average along integration path.
    """

    def __init__(self, model, steps=16, **kwargs):

        subanalyzer_kwargs = {}
        kwargs_keys = ["postprocess"]
        for key in kwargs_keys:
            if key in kwargs:
                subanalyzer_kwargs[key] = kwargs.pop(key)
        subanalyzer = Gradient(model, **subanalyzer_kwargs)

        super(IntegratedGradients, self).__init__(subanalyzer,
                                                  steps=steps,
                                                  **kwargs)


###############################################################################
###############################################################################
###############################################################################

class SmoothGrad(wrapper.GaussianSmoother):
    """Smooth grad analyzer.

    Applies the "smooth grad" algorithm to analyze the model.

    :param model: A Keras model.
    :param augment_by_n: Number of distortions to average for smoothing.
    """

    def __init__(self, model, augment_by_n=16, **kwargs):
        subanalyzer_kwargs = {}
        kwargs_keys = ["postprocess"]
        for key in kwargs_keys:
            if key in kwargs:
                subanalyzer_kwargs[key] = kwargs.pop(key)
        subanalyzer = Gradient(model, **subanalyzer_kwargs)

        super(SmoothGrad, self).__init__(subanalyzer,
                                         augment_by_n=augment_by_n,
                                         **kwargs)
