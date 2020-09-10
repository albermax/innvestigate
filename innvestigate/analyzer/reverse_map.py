import warnings
import tensorflow as tf

###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.backend as K
import numpy as np
from ..utils.keras import graph as kgraph

import tensorflow.keras.layers as keras_layers

class ReplacementLayer():
    """
    Base class for providing explainability functionality.

    This class wraps a single network layer, providing hooks for applying the layer and retrieving an explanation.
    Basically:
    * Any forward passes required for computing the explanation are defined in apply method. During the forward pass, a callback is given to all child layers to retrieve their explanations
    * Wrappers (e.g., a GradientTape) around the forward pass(es) that are required to compute an explanation can be defined and returned by wrap_hook method
    * In wrap_hook method, forward pass(es) are applied by calling on apply method
    * Explanation is computed in explain_hook method and then passed to callback functions of parent ReplacementLayers

    :param layer: Layer (of base class tensorflow.keras.layers.Layer) of to wrap
    :param layer_next: List of Layers in the network that receive output of layer (=child layers)

    This is just a base class. To extend this for specific XAI methods,
    - apply()
    - wrap_hook()
    - explain_hook()
    should be overwritten accordingly

    """
    def __init__(self, layer, layer_next=[], head_mapping=None):

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
        """
        callback function called by child layers when their explanation is computed.

        * aggregates explanations of all children
        * calls explain_hook to compute own explanation
        * sends own explanation to all parent layers by calling their callback functions

        :param reversed_outs: the child layer's explanation. None if this is the last layer.
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

            #print(self.name, np.shape(input_vals), np.shape(rev_outs), np.shape(self.hook_vals[0]))
            self.explanation = self.explain_hook(input_vals, rev_outs, self.hook_vals)

            # callbacks
            if self.callbacks is not None:
                # check if multiple inputs explained
                if len(self.callbacks) > 1 and not isinstance(self.explanation, list):
                    raise ValueError(self.name + ": This layer has " + str(
                        len(self.callbacks)) + " inputs, but no list of explanations was provided.")
                elif len(self.callbacks) > 1 and len(self.callbacks) != len(self.explanation):
                    raise ValueError(
                        self.name + ": This layer has " + str(len(self.callbacks)) + " inputs, but only " + str(
                            len(self.explanation)) + " explanations were computed")

                if len(self.callbacks) > 1:
                    for c, callback in enumerate(self.callbacks):
                        callback(self.explanation[c])
                else:
                    self.callbacks[0](self.explanation)

            # reset
            self.input_vals = None
            self.reversed_output_vals = None
            self.callbacks = None
            self.hook_vals = None

    def _forward(self, Ys, neuron_selection=None, stop_mapping_at_layers=None):
        """
        Forward Pass to all child layers
        * If this is the last layer, directly calls try_explain to compute own explanation
        * Otherwise calls try_apply on all child layers

        :param Ys: output of own forward pass
        :param neuron_selection: neuron_selection parameter (see try_apply)
        :param stop_mapping_at_layers: stop_mapping_at_layers parameter (see try_apply)
        """
        if len(self.layer_next) == 0 or self.name in stop_mapping_at_layers:
            # last layer: directly compute explanation
            self.try_explain(None)
        else:
            # forward
            for layer_n in self.layer_next:
                layer_n.try_apply(Ys, self.try_explain, neuron_selection, stop_mapping_at_layers)

    #@tf.function #this is not faster
    def _neuron_select(self, Ys, neuron_selection):
        """
        Performs neuron_selection on Ys

        :param Ys: output of own forward pass
        :param neuron_selection: neuron_selection parameter (see try_apply)
        """
        #error handling is done before, in try_apply

        if neuron_selection is None:
            Ys = Ys
        elif isinstance(neuron_selection, tf.Tensor):
            Ys = tf.gather_nd(Ys, neuron_selection, batch_dims=1)
        else:
            Ys = K.max(Ys, axis=1, keepdims=True)
        return Ys

    def try_apply(self, ins, callback=None, neuron_selection=None, stop_mapping_at_layers=None):
        """
        Tries to apply own forward pass:
        * Aggregates inputs and callbacks of all parent layers
        * Performs a canonization  of the neuron_selection parameter
        * Calls wrap_hook (wrapped forward pass(es))
        * Calls _forward (forward result of forward pass to child layers)

        :param ins output of own forward pass
        :param neuron_selection: neuron_selection parameter. One of the following:
            - None or "all"
            - "max_activation"
            - int
            - list or np.array of int, with length equal to batch size
        :param stop_mapping_at_layers: list of layers to stop mapping at
        :param callback callback function of the parent layer that called self.try_apply
        """
        # DEBUG
        # print(self.name, self.input_shape, np.shape(ins))

        # aggregate inputs
        if self.input_vals is None:
            self.input_vals = []
        self.input_vals.append(ins)

        # aggregate callbacks
        if callback is not None:
            if self.callbacks is None:
                self.callbacks = []
            self.callbacks.append(callback)

        # reset explanation
        self.explanation = None

        # apply layer only if all inputs collected. Then reset inputs
        if len(self.input_vals) == len(self.input_shape):

            # tensorify wrap_hook inputs as much as possible for graph efficiency
            input_vals = self.input_vals
            if len(input_vals) == 1:
                input_vals = input_vals[0]

            # adapt neuron_selection param for max efficiency
            if len(self.layer_next) == 0 or self.name in stop_mapping_at_layers:
                if neuron_selection is None:
                    neuron_selection_tmp = None
                elif isinstance(neuron_selection, str) and neuron_selection == "all":
                    neuron_selection_tmp = None
                elif isinstance(neuron_selection, str) and neuron_selection == "max_activation":
                    neuron_selection_tmp = "max_activation"
                elif isinstance(neuron_selection, int):
                    #if len(self.output_shape) > 1 or len(self.output_shape[0]) > 2:
                    #    raise ValueError("Expected last layer " + self.name + " to have only one output with shape dimension 2, but got " + str(self.output_shape))
                    #else:
                    neuron_selection_tmp = [[neuron_selection] for n in range(self.input_vals[0].shape[0])]
                    neuron_selection_tmp = tf.constant(neuron_selection_tmp)
                elif isinstance(neuron_selection, list) or (
                        hasattr(neuron_selection, "shape") and len(neuron_selection.shape) == 1):
                    #if len(self.output_shape) > 1 or len(self.output_shape[0]) > 2:
                    #    raise ValueError("Expected last layer " + self.name + " to have only one output with shape dimension 2, but got " + str(self.output_shape))
                    #elif len(np.shape(neuron_selection)) < 1:
                    #    raise ValueError("Expected parameter neuron_selection to have only one dimension, but got neuron_selection of shape " + str(np.shape(neuron_selection)))
                    #else:
                    neuron_selection_tmp = [[n] for n in neuron_selection]
                    neuron_selection_tmp = tf.constant(neuron_selection_tmp)
                else:
                    raise ValueError(
                        "Parameter neuron_selection only accepts the following values: None, 'all', 'max_activation', <int>, <list>, <one-dimensional array>")
            else:
                neuron_selection_tmp = neuron_selection

            # apply and wrappers
            self.hook_vals = self.wrap_hook(input_vals, neuron_selection_tmp, stop_mapping_at_layers)

            # forward
            if isinstance(self.hook_vals, tuple):
                self._forward(self.hook_vals[0], neuron_selection, stop_mapping_at_layers)
            else:
                self._forward(self.hook_vals, neuron_selection, stop_mapping_at_layers)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers):
        """
        hook that wraps and applies the layer function.
        E.g., by defining a GradientTape
        * should contain a call to self._neuron_select.
        * may define any wrappers around

        :param ins: input(s) of this layer
        :param neuron_selection: neuron_selection parameter (see try_apply)
        :param neuron_selection: stop_mapping_at_layers parameter (see try_apply)

        :returns output of layer function + any wrappers that were defined and are needed in explain_hook

        To be extended for specific XAI methods
        """
        outs = self.layer_func(ins)

        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or self.name in stop_mapping_at_layers:
            outs = self._neuron_select(outs, neuron_selection)

        return outs

    def explain_hook(self, ins, reversed_outs, args):
        """
        hook that computes the explanations.
        * Core XAI functionality

        :param ins: input(s) of this layer
        :param args: outputs of wrap_hook (any parameters that may be needed to compute explanation)
        :param reversed_outs: either backpropagated explanation(s) of child layers, or None if this is the last layer

        :returns explanation, or tensor of multiple explanations if the layer has multiple inputs (one for each)

        To be extended for specific XAI methods
        """
        outs = args

        if reversed_outs is None:
            reversed_outs = outs

        if len(self.layer_next) > 1:
            #TODO is this addition correct?
            ret = keras_layers.Add(dtype=tf.float32)([r for r in reversed_outs])
        elif len(self.input_shape) > 1:
            ret = [reversed_outs for i in self.input_shape]
            ret = tf.keras.layers.concatenate(ret, axis=1)
        else:
            ret = reversed_outs
        return ret


class GradientReplacementLayer(ReplacementLayer):
    """
    Simple extension of ReplacementLayer
    * Explains by computing gradients of outputs w.r.t. inputs of layer
    """
    def __init__(self, *args, **kwargs):
        super(GradientReplacementLayer, self).__init__(*args, **kwargs)

    def wrap_hook(self, ins, neuron_selection, stop_mapping_at_layers):
        #print("WRAP ", ins.shape, neuron_selection.shape)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0 or self.name in stop_mapping_at_layers:
                outs = self._neuron_select(outs, neuron_selection)

        return outs, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, tape = args

        if reversed_outs is None:
            reversed_outs = outs

        # correct number of outs
        if len(self.layer_next) > 1:
            outs = [outs for l in self.layer_next]

        if len(self.layer_next) > 1:
            if len(self.input_shape) > 1:
                ret = [keras_layers.Add(dtype=tf.float32)([tape.gradient(o, i, output_gradients=r) for o, r in zip(outs, reversed_outs)]) for i in ins]
            else:
                ret = keras_layers.Add(dtype=tf.float32)([tape.gradient(o, ins, output_gradients=r) for o, r in zip(outs, reversed_outs)])
        else:
            if len(self.input_shape) > 1:
                ret = [tape.gradient(outs, i, output_gradients=reversed_outs) for i in ins]
            else:
                ret = tape.gradient(outs, ins, output_gradients=reversed_outs)

        return ret


def reverse_map(
        model,
        reverse_mappings,
        default_reverse_mapping,
        ):
    """
    Builds the reverse_map by wrapping network layer(s) into ReplacementLayer(s)

    :param model: model to be analyzed
    :param reverse_mappings: mapping layer->reverse mapping (ReplacementLayer or some subclass thereof)
    :param default_reverse_mapping: ReplacementLayer or some subclass thereof; default mapping to use

    :returns reversed "model" as a list of input layers and a list of wrapped layers
    """

    #build model that is to be analyzed
    layers = kgraph.get_model_layers(model)

    # set all replacement layers
    replacement_layers = []
    for layer in layers:
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
    for i, t in enumerate(model.inputs):
        for layer in replacement_layers:
            if id(layer.layer_func.output) == id(t):
                input_layers.append(layer)
        if len(input_layers) < i+1:
            #if we did not append an input layer, we need to create one
            #TODO case for no input layer here
            raise ValueError("Temporary error. You need to explicitly define an Input Layer for now")

    return input_layers, replacement_layers

def apply_reverse_map(Xs, reverse_ins, reverse_layers, neuron_selection="max_activation", layer_names=None, stop_mapping_at_layers=[]):
    """
    Computes an explanation by applying a reversed model

    :param Xs: tensor or np.array of Input to be explained. Shape (n_ins, batch_size, ...) in model has multiple inputs, or (batch_size, ...) otherwise
    :param reverse_ins: list of input ReplacementLayer(s)
    :param reverse_layers: list of all ReplacementLayer(s) denoting the reversed model
    :param neuron_selection: neuron_selection parameter. Used to only compute explanation w.r.t. specific output neurons. One of the following:
            - None or "all"
            - "max_activation"
            - int
            - list or np.array of int, with length equal to batch size
    :param layer_names: None or list of layer names whose explanations should be returned.
                        Can be used to obtain intermediate explanations or explanations of multiple layers
    :param stop_mapping_at_layers: list of layers to stop mapping at ("output" layers)

    :returns list of explanations. Each explanation in this list (np.array) corresponds to one layer.
             The list either contains the explanations of all input layers if layer_names=None, or the explanations for all layers in layer_names otherwise.
    """
    #shape of Xs: (n_ins, batch_size, ...), or (batch_size, ...)

    warnings.simplefilter("always")
    if len(stop_mapping_at_layers) > 0 and (isinstance(neuron_selection, int) or isinstance(neuron_selection, list) or isinstance(neuron_selection, np.ndarray)):
        warnings.warn("You are specifying layers to stop forward pass at, and also neuron-selecting by index. Please make sure the corresponding shapes fit together!")

    if not isinstance(Xs, tf.Tensor):
        try:
            Xs = tf.constant(Xs)
        except:
            raise ValueError("Xs has not supported type ", type(Xs))

    #format input & obtain explanations
    if len(reverse_ins) == 1:
        #single input network
        reverse_ins[0].try_apply(tf.constant(Xs), neuron_selection=neuron_selection, stop_mapping_at_layers=stop_mapping_at_layers)

    else:
        #multiple inputs. reshape to (n_ins, batch_size, ...)
        for i, reverse_in in enumerate(reverse_ins):
            reverse_in.try_apply(tf.constant(Xs[i]), neuron_selection=neuron_selection, stop_mapping_at_layers=stop_mapping_at_layers)

    #obtain explanations for specified layers
    if layer_names is None:
        #just explain input layers
        hm = [layer.explanation.numpy() for layer in reverse_ins]
    else:
        hm = []
        for name in layer_names:
            layer = [layer for layer in reverse_layers if layer.name==name][0]
            hm.append(layer.explanation.numpy())

    return hm

