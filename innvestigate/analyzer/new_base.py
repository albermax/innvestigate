from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


from . import base
from . import wrapper
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph


# Get Python six functionality:
from builtins import zip
import six

###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.backend as K
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models
import numpy as np
import warnings


from .. import layers as ilayers
from .. import utils as iutils
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph


__all__ = [
    "NotAnalyzeableModelException",
    "AnalyzerBase",

    "TrainerMixin",
    "OneEpochTrainerMixin",

    "AnalyzerNetworkBase",
    "ReverseAnalyzerBase"
]


###############################################################################
###############################################################################
###############################################################################


class NotAnalyzeableModelException(Exception):
    """Indicates that the model cannot be analyzed by an analyzer."""
    pass


class AnalyzerBase(object):
    """ The basic interface of an iNNvestigate analyzer.

    This class defines the basic interface for analyzers:

    >>> model = create_keras_model()
    >>> a = Analyzer(model)
    >>> a.fit(X_train)  # If analyzer needs training.
    >>> analysis = a.analyze(X_test)
    >>>
    >>> state = a.save()
    >>> a_new = A.load(*state)
    >>> analysis = a_new.analyze(X_test)

    :param model: A Keras model.
    :param disable_model_checks: Do not execute model checks that enforce
      compatibility of analyzer and model.

    .. note:: To develop a new analyzer derive from
      :class:`AnalyzerNetworkBase`.
    """

    def __init__(self, model, disable_model_checks=False):
        self._model = model
        self._disable_model_checks = disable_model_checks

        self._do_model_checks()

    def _add_model_check(self, check, message, check_type="exception"):
        if getattr(self, "_model_check_done", False):
            raise Exception("Cannot add model check anymore."
                            " Check was already performed.")

        if not hasattr(self, "_model_checks"):
            self._model_checks = []

        check_instance = {
            "check": check,
            "message": message,
            "type": check_type,
        }
        self._model_checks.append(check_instance)

    def _do_model_checks(self):
        model_checks = getattr(self, "_model_checks", [])

        if not self._disable_model_checks and len(model_checks) > 0:
            check = [x["check"] for x in model_checks]
            types = [x["type"] for x in model_checks]
            messages = [x["message"] for x in model_checks]

            checked = kgraph.model_contains(self._model, check)
            tmp = zip(iutils.to_list(checked), messages, types)

            for checked_layers, message, check_type in tmp:
                if len(checked_layers) > 0:
                    tmp_message = ("%s\nCheck triggerd by layers: %s" %
                                   (message, checked_layers))

                    if check_type == "exception":
                        raise NotAnalyzeableModelException(tmp_message)
                    elif check_type == "warning":
                        # TODO(albermax) only the first warning will be shown
                        warnings.warn(tmp_message)
                    else:
                        raise NotImplementedError()

        self._model_check_done = True

    def fit(self, *args, **kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        disable_no_training_warning = kwargs.pop("disable_no_training_warning",
                                                 False)
        if not disable_no_training_warning:
            # issue warning if not training is foreseen,
            # but is fit is still called.
            warnings.warn("This analyzer does not need to be trained."
                          " Still fit() is called.", RuntimeWarning)

    def fit_generator(self, *args, **kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        disable_no_training_warning = kwargs.pop("disable_no_training_warning",
                                                 False)
        if not disable_no_training_warning:
            # issue warning if not training is foreseen,
            # but is fit is still called.
            warnings.warn("This analyzer does not need to be trained."
                          " Still fit_generator() is called.", RuntimeWarning)

    def analyze(self, X):
        """
        Analyze the behavior of model on input `X`.

        :param X: Input as expected by model.
        """
        raise NotImplementedError()


###############################################################################
###############################################################################
###############################################################################


class TrainerMixin(object):
    """Mixin for analyzer that adapt to data.

    This convenience interface exposes a Keras like training routing
    to the user.
    """

    # todo: extend with Y
    def fit(self,
            X=None,
            batch_size=32,
            **kwargs):
        """
        Takes the same parameters as Keras's :func:`model.fit` function.
        """
        generator = iutils.BatchSequence(X, batch_size)
        return self._fit_generator(generator,
                                  **kwargs)

    def fit_generator(self, *args, **kwargs):
        """
        Takes the same parameters as Keras's :func:`model.fit_generator`
        function.
        """
        return self._fit_generator(*args, **kwargs)

    def _fit_generator(self,
                       generator,
                       steps_per_epoch=None,
                       epochs=1,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0,
                       disable_no_training_warning=None):
        raise NotImplementedError()


class OneEpochTrainerMixin(TrainerMixin):
    """Exposes the same interface and functionality as :class:`TrainerMixin`
    except that the training is limited to one epoch.
    """

    def fit(self, *args, **kwargs):
        """
        Same interface as :func:`fit` of :class:`TrainerMixin` except that
        the parameter epoch is fixed to 1.
        """
        return super(OneEpochTrainerMixin, self).fit(*args, epochs=1, **kwargs)

    def fit_generator(self, *args, **kwargs):
        """
        Same interface as :func:`fit_generator` of :class:`TrainerMixin` except that
        the parameter epoch is fixed to 1.
        """
        steps = kwargs.pop("steps", None)
        return super(OneEpochTrainerMixin, self).fit_generator(
            *args,
            steps_per_epoch=steps,
            epochs=1,
            **kwargs)


###############################################################################
###############################################################################
###############################################################################


class AnalyzerNetworkBase(AnalyzerBase):
    """Convenience interface for analyzers.

    This class provides helpful functionality to create analyzer's.
    Basically it:

    * takes the input model and adds a layer that selects
      the desired output neuron to analyze.
    * passes the new model to :func:`_create_analysis` which should
      return the analysis as Keras tensors.
    * compiles the function and serves the output to :func:`analyze` calls.
    * allows :func:`_create_analysis` to return tensors
      that are intercept for debugging purposes.

    :param neuron_selection_mode: How to select the neuron to analyze.
      Possible values are 'max_activation', 'index' for the neuron
      (expects indices at :func:`analyze` calls), 'all' take all neurons.
    :param allow_lambda_layers: Allow the model to contain lambda layers.
    """

    def __init__(self, model,
                 neuron_selection_mode="max_activation",
                 allow_lambda_layers=False,
                 **kwargs):
        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection parameter is not valid.")
        self._neuron_selection_mode = neuron_selection_mode

        self._allow_lambda_layers = allow_lambda_layers
        self._add_model_check(
            lambda layer: (not self._allow_lambda_layers and
                           isinstance(layer, keras_layers.Lambda)),
            ("Lamda layers are not allowed. "
             "To force use set allow_lambda_layers parameter."),
            check_type="exception",
        )

        self._special_helper_layers = []

        super(AnalyzerNetworkBase, self).__init__(model, **kwargs)

    def _add_model_softmax_check(self):
        """
        Adds check that prevents models from containing a softmax.
        """
        self._add_model_check(
            lambda layer: kchecks.contains_activation(
                layer, activation="softmax"),
            "This analysis method does not support softmax layers.",
            check_type="exception",
        )

    def create_analyzer_model(self):
        """
        Creates the analyze functionality. If not called beforehand
        it will be called by :func:`analyze`.
        """

        self._analyzer_model = self._create_analysis(self._model)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        """
        Interface that needs to be implemented by a derived class.

        This function is expected to create a Keras graph that creates
        a custom analysis for the model inputs given the model outputs.

        :param model: Target of analysis.
        :param stop_analysis_at_tensors: A list of tensors where to stop the
          analysis. Similar to stop_gradient arguments when computing the
          gradient of a graph.
        :return: Either one-, two- or three-tuple of lists of tensors.
          * The first list of tensors represents the analysis for each
            model input tensor. Tensors present in stop_analysis_at_tensors
            should be omitted.
          * The second list, if present, is a list of debug tensors that will
            be passed to :func:`_handle_debug_output` after the analysis
            is executed.
          * The third list, if present, is a list of constant input tensors
            added to the analysis model.
        """
        raise NotImplementedError()

    def _handle_debug_output(self, debug_values):
        raise NotImplementedError()

    def analyze(self, X, neuron_selection=None, layer_names=None):
        """
                Same interface as :class:`Analyzer` besides

                :param neuron_selection: If neuron_selection_mode is 'index' this
                  should be an integer with the index for the chosen neuron.
                """
        #TODO: check X, neuron_selection, and layer_selection for validity
        if not hasattr(self, "_analyzer_model"):
            self.create_analyzer_model()
        inp, all = self._analyzer_model
        ret = apply_reverse_map(X, inp, all, neuron_selection=neuron_selection, layer_names=layer_names)
        ret = self._postprocess_analysis(ret)

        return ret

    def _postprocess_analysis(self, X):
        return X

def gradient_reverse_map(
    #Alternative to kgraph.reverse_model.
        model,
        reverse_mappings,
        default_reverse_mapping,
        head_mapping,
        stop_mapping_at_tensors,
        verbose=False):
    #TODO: verbose
    #TODO: HeadMapping
    #TODO this is just the basic core. Add full functionality of kgraph.reverse_model
    stop_mapping_at_tensors = [x.name.split(":")[0] for x in stop_mapping_at_tensors]

    layers = kgraph.get_model_layers(model)
    replacement_layers = []

    #set all replacement layers
    for layer in layers:
        if not layer.name in stop_mapping_at_tensors:
            layer_next = []
            wrapper_class = reverse_mappings(layer)
            if wrapper_class is None:
                wrapper_class = default_reverse_mapping(layer)

            if not issubclass(wrapper_class, ReplacementLayer):
                raise ValueError("Reverse Mappings should be an instance of ReplacementLayer")

            replacement_layers.append(wrapper_class(layer, layer_next))

    #connect graph structure
    for layer in replacement_layers:
        for layer2 in replacement_layers:
            inp = layer2.layer_func.input
            out = layer.layer_func.output
            if not isinstance(inp, list):
                inp = [inp]
            if not isinstance(out, list):
                out = [out]

            for i in inp:
                if id(i) in [id(o) for o in out] and id(layer) != id(layer2):
                    layer.layer_next.append(layer2)

    #find input access points
    input_layers = []
    for t in model.inputs:
        for layer in replacement_layers:
            if id(layer.layer_func.output) == id(t):
                input_layers.append(layer)

    #TODO rethink this format. probably make a ReplacementModel class or something
    return input_layers, replacement_layers

def apply_reverse_map(Xs, reverse_ins, reverse_layers, neuron_selection=None, layer_names=None):
    #shape of Xs: (batch_size, n_ins, ...), or (batch_size, ...)

    #format input & obtain explanations
    if len(reverse_ins) == 1:
        #single input network
        reverse_ins[0].try_apply(tf.constant(Xs), neuron_selection=neuron_selection)

    else:
        #multiple inputs. reshape to (n_ins, batch_size, ...)
        Xs_new = [[X[i] for X in Xs] for i, _ in enumerate(reverse_ins)]
        Xs = Xs_new
        for i, reverse_in in enumerate(reverse_ins):
            reverse_in.try_apply(tf.constant(Xs[i]), neuron_selection=neuron_selection)

    #obtain explanations for specified layers
    if layer_names is None:
        #just explain input layers
        hm = [layer.explanation for layer in reverse_ins]
    else:
        hm = []
        for name in layer_names:
            layer = [layer for layer in reverse_layers if layer.name==name][0]
            hm.append(layer.explanation)

    return hm

def GuidedBackpropReverseReLU(Xs, Ys, reversed_Ys, tape):
    activation = keras_layers.Activation("relu")
    reversed_Ys = kutils.apply(activation, reversed_Ys)
    return tape.gradient(Ys, Xs, output_gradients=reversed_Ys)

#TODO: more replacement layers
class ReplacementLayer():
    #TODO: consider merge layers. possibly need to redesign workflow. something asynchroneous?
    #TODO: consider making this a sub_class of keras.Layer?
    def __init__(self, layer, layer_next=[]):

        self.layer_func = layer
        self.layer_next = layer_next
        self.name = layer.name

        self.input_shape = layer.input_shape
        if not isinstance(self.input_shape, list):
            self.input_shape = [self.input_shape]
        self.output_shape = layer.output_shape
        if not isinstance(self.output_shape, list):
            self.output_shape = [self.output_shape]

        self.input_vals = None
        self.reversed_output_vals = None
        self.callbacks = None
        self.hook_vals = None
        self.explanation = None

    def try_explain(self, reversed_outs):
        #aggregate explanations
        if reversed_outs is not None:
            if self.reversed_output_vals is None:
                self.reversed_output_vals = []
            self.reversed_output_vals.append(reversed_outs)

        #last layer or aggregation finished
        if self.reversed_output_vals is None or len(self.reversed_output_vals) == len(self.layer_next):
            # apply post hook: explain
            if self.hook_vals is None:
                raise ValueError("self.hook_vals should contain values at this point. Is self.wrap_hook working correctly?")
            input_vals = self.input_vals
            if len(input_vals) == 1:
                input_vals = input_vals[0]
            self.explanation = self.explain_hook(input_vals, self.reversed_output_vals, self.hook_vals)

            # callbacks
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self.explanation)

            #reset
            self.input_vals = None
            self.reversed_output_vals = None
            self.callbacks = None
            self.hook_vals = None

    def _forward(self, Ys, neuron_selection=None):
        if len(self.layer_next) == 0:
            #last layer: directly compute explanation
            self.try_explain(None)
        else:
            #forward
            for layer_n in self.layer_next:
                layer_n.try_apply(Ys, neuron_selection, self.try_explain)

    def _neuron_select(self, Ys, neuron_selection):

        if neuron_selection is None or neuron_selection == "all":
            Ys = Ys
        elif neuron_selection == "max_activation":
            Ys = K.max(Ys, axis=-1, keepdims=True)
        elif isinstance(neuron_selection, int):
            # TODO error handling
            neuron_selection = [[neuron_selection] for n in range(Ys.shape[0])]
            Ys = tf.gather_nd(Ys, neuron_selection, batch_dims=1)
        elif isinstance(neuron_selection, list):
            #TODO error handling
            #TODO this assumes that the last layer has shape (batch_size, n); is that a valid assumption?
            if len(np.shape(neuron_selection)) < 2:
                neuron_selection = [[n] for n in neuron_selection]
            Ys = tf.gather_nd(Ys, neuron_selection, batch_dims=1)
        else:
            raise ValueError(
                "Parameter neuron_selection only accepts the following values: None, 'all', 'max_activation', <int>, <list>")
        return Ys

    def try_apply(self, ins, neuron_selection=None, callback=None):
        #aggregate inputs
        if self.input_vals is None:
            self.input_vals = []
        self.input_vals.append(ins)

        #aggregate callbacks
        if callback is not None:
            if self.callbacks is None:
                self.callbacks = []
            self.callbacks.append(callback)

        #reset explanation
        self.explanation = None

        #apply layer only if all inputs collected. Then reset inputs
        if len(self.input_vals) == len(self.input_shape):
            # apply layer
            input_vals = self.input_vals
            if len(input_vals) == 1:
                input_vals = input_vals[0]
            self.hook_vals = self.wrap_hook(input_vals, neuron_selection)

            #forward
            self._forward(self.hook_vals[0], neuron_selection)

    def wrap_hook(self, ins, neuron_selection):
        """
        hook that wraps the layer function. should contain a call to layer_func and _neuron_select.
        """
        outs = self.layer_func(ins)

        #check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0:
            outs = self._neuron_select(outs, neuron_selection)

        return outs

    def explain_hook(self, ins, reversed_outs, args):
        return reversed_outs

class GradientReplacementLayer(ReplacementLayer):
    def __init__(self, *args, **kwargs):
        super(GradientReplacementLayer, self).__init__(*args, **kwargs)

    def wrap_hook(self, ins, neuron_selection):
        with tf.GradientTape() as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0:
                outs = self._neuron_select(outs, neuron_selection)

        return outs, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, tape = args
        ret = tape.gradient(outs, ins, output_gradients=reversed_outs)
        return ret

class ReverseAnalyzerBase(AnalyzerNetworkBase):
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

        #TODO set verbose correctly somewhere
        self._reverse_verbose = False

        super(ReverseAnalyzerBase, self).__init__(model, **kwargs)

    def _reverse_mapping(self, layer):
        """
        This function should return a reverse mapping for the passed layer.

        If this function returns None, :func:`_default_reverse_mapping`
        is applied.

        :param layer: The layer for which a mapping should be returned.
        :return: The mapping can be of the following forms:
          * A function of form (A) f(Xs, Ys, reversed_Ys, reverse_state)
            that maps reversed_Ys to reversed_Xs (which should contain
            tensors of the same shape and type).
          * A function of form f(B) f(layer, reverse_state) that returns
            a function of form (A).
          * A :class:`ReverseMappingBase` subclass.
        """
        if layer in self._special_helper_layers:
            # Special layers added by AnalyzerNetworkBase
            # that should not be exposed to user.
            return GradientReplacementLayer

        return self._apply_conditional_reverse_mappings(layer)

    def _add_conditional_reverse_mapping(
            self, condition, mapping, priority=-1, name=None):
        """
        This function should return a reverse mapping for the passed layer.

        If this function returns None, :func:`_default_reverse_mapping`
        is applied.

        :param condition: Condition when this mapping should be applied.
          Form: f(layer) -> bool
        :param mapping: The mapping can be of the following forms:
          * A function of form (A) f(Xs, Ys, reversed_Ys, reverse_state)
            that maps reversed_Ys to reversed_Xs (which should contain
            tensors of the same shape and type).
          * A function of form f(B) f(layer, reverse_state) that returns
            a function of form (A).
          * A :class:`ReverseMappingBase` subclass.
        :param priority: The higher the earlier the condition gets
          evaluated.
        :param name: An identifying name.
        """
        if getattr(self, "_reverse_mapping_applied", False):
            raise Exception("Cannot add conditional mapping "
                            "after first application.")

        if not hasattr(self, "_conditional_reverse_mappings"):
            self._conditional_reverse_mappings = {}

        if priority not in self._conditional_reverse_mappings:
            self._conditional_reverse_mappings[priority] = []

        tmp = {"condition": condition, "mapping": mapping, "name": name}
        self._conditional_reverse_mappings[priority].append(tmp)

    def _apply_conditional_reverse_mappings(self, layer):
        mappings = getattr(self, "_conditional_reverse_mappings", {})
        self._reverse_mapping_applied = True

        # Search for mapping. First consider ones with highest priority,
        # inside priority in order of adding.
        sorted_keys = sorted(mappings.keys())[::-1]
        for key in sorted_keys:
            for mapping in mappings[key]:
                if mapping["condition"](layer):
                    return mapping["mapping"]

        return None

    def _default_reverse_mapping(self, layer):
        """
        Fallback function to map reversed_Ys to reversed_Xs
        (which should contain tensors of the same shape and type).
        """
        return GradientReplacementLayer

    def _head_mapping(self):
        """
        Map output tensors to new values before passing
        them into the reverted network.
        """
        return ReplacementLayer

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        inp, rep = gradient_reverse_map(
            model,
            reverse_mappings=self._reverse_mapping,
            default_reverse_mapping=self._default_reverse_mapping,
            head_mapping=self._head_mapping,
            stop_mapping_at_tensors=stop_analysis_at_tensors,
            verbose=self._reverse_verbose,
        )

        return inp, rep