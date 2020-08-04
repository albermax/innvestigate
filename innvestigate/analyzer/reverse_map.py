
import tensorflow as tf

###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.backend as K
import numpy as np
from ..utils.keras import graph as kgraph

import tensorflow.keras.layers as keras_layers

class ReplacementLayer():
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

    def _forward(self, Ys, neuron_selection=None):
        if len(self.layer_next) == 0:
            # last layer: directly compute explanation
            self.try_explain(None)
        else:
            # forward
            for layer_n in self.layer_next:
                layer_n.try_apply(Ys, neuron_selection, self.try_explain)

    #@tf.function #this is not faster
    def _neuron_select(self, Ys, neuron_selection):
        #error handling is done before, in try_apply
        if isinstance(neuron_selection, tf.Tensor):
            Ys = tf.gather_nd(Ys, neuron_selection, batch_dims=1)
        elif neuron_selection is None:
            Ys = Ys
        else:
            Ys = K.max(Ys, axis=-1, keepdims=True)
        return Ys

    def try_apply(self, ins, neuron_selection=None, callback=None):

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
            if len(self.layer_next) != 0:
                # basically just a filler value (we are not in the last layer)
                # allowing for efficient graph building
                neuron_selection_tmp = tf.constant(0)
            else:
                if neuron_selection is None:
                    neuron_selection_tmp = None
                elif isinstance(neuron_selection, str) and neuron_selection == "all":
                    neuron_selection_tmp = None
                elif isinstance(neuron_selection, str) and neuron_selection == "max_activation":
                    neuron_selection_tmp = "max_activation"
                elif isinstance(neuron_selection, int):
                    if len(self.output_shape) > 1 or len(self.output_shape[0]) > 2:
                        raise ValueError("Expected last layer " + self.name + "to have only one output with shape dimension 2, but got " + str(self.output_shape))
                    else:
                        neuron_selection_tmp = [[neuron_selection] for n in range(self.input_vals[0].shape[0])]
                        neuron_selection_tmp = tf.constant(neuron_selection_tmp)
                elif isinstance(neuron_selection, list) or (
                        hasattr(neuron_selection, "shape") and len(neuron_selection.shape) == 1):
                    # TODO this assumes that the last layer has shape (batch_size, n); is that a valid assumption?
                    if len(self.output_shape) > 1 or len(self.output_shape[0]) > 2:
                        raise ValueError("Expected last layer " + self.name + "to have only one output with shape dimension 2, but got " + str(self.output_shape))
                    elif len(np.shape(neuron_selection)) < 1:
                        raise ValueError("Expected parameter neuron_selection to have only one dimension, but got neuron_selection of shape " + str(np.shape(neuron_selection)))
                    else:
                        neuron_selection_tmp = [[n] for n in neuron_selection]
                        neuron_selection_tmp = tf.constant(neuron_selection_tmp)
                else:
                    raise ValueError(
                        "Parameter neuron_selection only accepts the following values: None, 'all', 'max_activation', <int>, <list>, <one-dimensional array>")

            # apply layer. allow for
            self.hook_vals = self.wrap_hook(input_vals, neuron_selection_tmp)

            # forward
            if isinstance(self.hook_vals, tuple):
                self._forward(self.hook_vals[0], neuron_selection)
            else:
                self._forward(self.hook_vals, neuron_selection)

    #@tf.function #this is not faster
    def apply(self, ins, neuron_selection):
        """
        applies layer / forward tf ops.
        for efficiency, keep as tf.function
        """
        outs = self.layer_func(ins)

        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0:
            outs = self._neuron_select(outs, neuron_selection)

        return outs

    def wrap_hook(self, ins, neuron_selection):
        """
        hook that wraps the layer function. should contain a call to self.apply.
        """
        return self.apply(ins, neuron_selection)

    def explain_hook(self, ins, reversed_outs, args):
        """
        hook that computes the explanations.
        param args: additional parameters
        param reversed_outs: either backpropagated explanation, or None if last layer

        returns: explanation, or tensor of multiple explanations if the layer has multiple inputs (one for each)
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
    def __init__(self, *args, **kwargs):
        super(GradientReplacementLayer, self).__init__(*args, **kwargs)

    def wrap_hook(self, ins, neuron_selection):
        #print("WRAP ", ins.shape, neuron_selection.shape)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ins)
            outs = self.apply(ins, neuron_selection)

        return outs, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, tape = args

        if reversed_outs is None:
            reversed_outs = outs

        # correct number of outs
        if len(self.layer_next) > 1:
            outs = [outs for l in self.layer_next]

        if len(self.layer_next) > 1:
            #print(self.name, np.shape(outs), np.shape(reversed_outs))
            #TODO: is this correct?
            keras_layers.Add(dtype=tf.float32)([tape.gradient(o, ins, output_gradients=r) for o, r in zip(outs, reversed_outs)])
            #raise ValueError("This basic function is not defined for layers with multiple children")
        if len(self.input_shape) > 1:
            ret = [tape.gradient(outs, i, output_gradients=reversed_outs) for i in ins]
        else:
            ret = tape.gradient(outs, ins, output_gradients=reversed_outs)
        return ret


def reverse_map(
    #Alternative to kgraph.reverse_model.
        model,
        reverse_mappings,
        default_reverse_mapping,
        stop_mapping_at_tensors,
        verbose=False):

    #TODO: verbose
    #TODO: HeadMapping
    #TODO this is just the basic core. Add full functionality of kgraph.reverse_model

    #build model that is to be analyzed
    layers = kgraph.get_model_layers(model)
    stop_mapping_at_tensors = [x.name.split(":")[0] for x in stop_mapping_at_tensors]

    # set all replacement layers
    replacement_layers = []
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

    return input_layers, replacement_layers

def apply_reverse_map(Xs, reverse_ins, reverse_layers, neuron_selection=None, layer_names=None):
    #shape of Xs: (n_ins, batch_size, ...), or (batch_size, ...)

    #TODO: output shape?
    #Returns: Explanation of Form (n_inputs, batch_size, ...)

    #format input & obtain explanations
    if len(reverse_ins) == 1:
        #single input network
        reverse_ins[0].try_apply(tf.constant(Xs), neuron_selection=neuron_selection)

    else:
        #multiple inputs. reshape to (n_ins, batch_size, ...)
        #Xs_new = [[X[i] for X in Xs] for i, _ in enumerate(reverse_ins)]
        #Xs = Xs_new
        for i, reverse_in in enumerate(reverse_ins):
            reverse_in.try_apply(tf.constant(Xs[i]), neuron_selection=neuron_selection)

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

