import warnings
warnings.simplefilter("ignore")
#warnings.simplefilter("always")
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

###############################################################################
###############################################################################
###############################################################################

import tensorflow.keras.backend as K
import numpy as np
from ..utils.keras import graph as kgraph
from ..utils.keras import functional as kfunctional

import tensorflow.keras.layers as keras_layers

#---------------------------------------------------Classes------------------------------------

class ReplacementLayer():
    """
    Base class for providing explainability functionality.

    This class wraps a single network layer, providing hooks for applying the layer and retrieving an explanation.
    Basically:
    * Any forward passes required for computing the explanation are defined in apply method. During the forward pass, a callback is given to all child layers to retrieve their explanations
    * Wrappers (e.g., a GradientTape) around the forward pass(es) that are required to compute an explanation can be defined and returned by wrap_hook method
    * In wrap_hook method, forward pass(es) are applied and Tapes defined
    * Explanation is computed in compute_explanation method and then passed to callback functions of parent ReplacementLayers

    :param layer: Layer (of base class tensorflow.keras.layers.Layer) of to wrap
    :param layer_next: List of Layers in the network that receive output of layer (=child layers)
    :param r_init_constant: If not None, defines a constant output mapping
    :param f_init_constant: If not None, defines a constant activation mapping

    This is just a base class. To extend this for specific XAI methods,
    - wrap_hook()
    - compute_explanation()
    should be overwritten accordingly

    """
    def __init__(self, layer, layer_next=[], r_init_constant=None, f_init_constant=None):

        #params
        self.layer_func = layer
        self.layer_next = layer_next
        self.name = layer.name
        self.r_init = r_init_constant
        self.f_init = f_init_constant

        #functions
        self._neuron_select = kfunctional.neuron_select
        self._out_func = None
        self._explain_func = None

        self.input_shape = layer.input_shape
        if not isinstance(self.input_shape, list):
            self.input_shape = [self.input_shape]
        self.output_shape = layer.output_shape
        if not isinstance(self.output_shape, list):
            self.output_shape = [self.output_shape]

        self.input_vals = None
        self.original_output_vals = None
        self.reversed_output_vals = None
        self.callbacks = None
        self.saved_forward_vals = None
        self.explanation = None

        ###############
        self.forward_after_stopping = False
        self.reached_after_stop_mapping = None
        self.activations_saved = False
        self.no_forward_pass = False
        self.debug = False

    def try_explain(self, reversed_outs):
        """
        callback function called by child layers when their explanation is computed.

        * aggregates explanations of all children
        * calls compute_explanation to compute own explanation
        * sends own explanation to all parent layers by calling their callback functions

        :param reversed_outs: the child layer's explanation. None if this is the last layer.
        """
        if self.debug == True:
            print("enter try_explain for", self.name)
        # aggregate explanations
        if reversed_outs is not None:
            if self.reversed_output_vals is None:
                self.reversed_output_vals = []
            self.reversed_output_vals.append(reversed_outs)

        # last layer or aggregation finished
        if self.reversed_output_vals is None or len(self.reversed_output_vals) == len(self.layer_next):
            # apply post hook: explain
            if self.saved_forward_vals is None:
                raise ValueError(
                    "self.saved_forward_vals should contain values at this point. Is self.wrap_hook working correctly?")
            input_vals = self.input_vals
            if len(input_vals) == 1:
                input_vals = input_vals[0]

            rev_outs = self.reversed_output_vals
            if rev_outs is not None:
                if len(rev_outs) == 1:
                    rev_outs = rev_outs[0]
            if self.debug == True:
                print("Backward at: ", self.name)
                print("layer_next:", len(self.layer_next))
                print(self.name, np.shape(input_vals), np.shape(rev_outs), np.shape(self.saved_forward_vals["outs"]))
            self.explanation = self.compute_explanation(input_vals, rev_outs)

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

            if self.no_forward_pass == True:
                # save activations
                self.activations_saved = True
            else:
                # reset
                self.input_vals = None
                self.saved_forward_vals = None
                self.callbacks = None

            self.reversed_output_vals = None

    def _forward(self, Ys, neuron_selection=None, stop_mapping_at_layers=None, r_init=None, f_init=None):
        """
        Forward Pass to all child layers
        * If this is the last layer, directly calls try_explain to compute own explanation
        * Otherwise calls try_apply on all child layers

        :param Ys: output of own forward pass
        :param neuron_selection: neuron_selection parameter (see try_apply)
        :param stop_mapping_at_layers: stop_mapping_at_layers parameter (see try_apply)
        :param r_init: None or Scalar or Array-Like or Dict {layer_name:scalar or array-like} reverse initialization value. Value with with explanation is initialized (i.e., head_mapping).
        :param f_init: None or Scalar or Array-Like or Dict {layer_name:scalar or array-like} forward initialization value. Value with which the forward is initialized.
        """
        if self.debug == True:
            print("Forward: ", self.name)
        if len(self.layer_next) == 0 :
            # last layer: directly compute explanation
            self.try_explain(None)
        elif stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers:
            self.try_explain(None)

            if self.forward_after_stopping:

                # if the output was mapped, this restores the original for the forwarding
                if self.original_output_vals is not None:
                    Ys = self.original_output_vals
                    self.original_output_vals = None

                # make a dummy callback so that logic does not get buggy
                def dummyCallback(reversed_outs):
                    pass

                for layer_n in self.layer_next:
                    layer_n.try_apply(Ys, dummyCallback, neuron_selection, stop_mapping_at_layers, r_init, f_init)


        else:
            # forward
            for layer_n in self.layer_next:
                layer_n.try_apply(Ys, self.try_explain, neuron_selection, stop_mapping_at_layers, r_init, f_init)

    @tf.custom_gradient
    def _toNumber(self, X, value):
        """
        Helper function to set a Tensor to a fixed value while having a gradient of "value"

        """

        y = tf.constant(value, dtype=tf.float32, shape=X.shape)

        def grad(dy, variables=None):  # variables=None and None as output necessary as toNumber requires two arguments

            return dy * tf.sign(X) * value, None

        return y, grad

    #@tf.function
    def _head_mapping(self, Ys, model_output_value=None):
        """
        Sets the model output to a fixed value. Used as initialization
        for the explanation method.

        :param model_output_value: output value of model / initialized value for explanation method

        """

        if model_output_value is not None:
            if isinstance(model_output_value, dict):
                if self.name in model_output_value.keys():
                    # model_output_value should be int or array-like. Shape should fit.
                    Ys = self._toNumber(Ys, model_output_value[self.name])
            else:
                # model_output_value should be int or array-like. Shape should fit.
                Ys = self._toNumber(Ys, model_output_value)

        return Ys

    def _neuron_sel_and_head_map(self, Ys, neuron_selection=None, model_output_value=None):

        #save original output
        self.original_output_vals = Ys

        #apply neuron selection and head mapping
        Ys = self._neuron_select(Ys, neuron_selection)
        Ys = self._head_mapping(Ys, model_output_value)

        return Ys

    def compute_output(self, ins, neuron_selection, stop_mapping_at_layers, r_init):
        """
        hook that wraps and applies the layer function.
        E.g., by defining a GradientTape
        * should contain a call to self._neuron_select.
        * may define any wrappers around

        :param ins: input(s) of this layer
        :param neuron_selection: neuron_selection parameter (see try_apply)
        :param stop_mapping_at_layers: None or stop_mapping_at_layers parameter (see try_apply)
        :param r_init: reverse initialization value. Value with with explanation is initialized (i.e., head_mapping).

        :returns output of layer function + any wrappers that were defined and are needed in compute_explanation

        To be extended for specific XAI methods
        """
        # check if final layer (i.e., no next layers)
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._out_func = kfunctional.final_out_func
        else:
            self._out_func = kfunctional.out_func

        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            outs = self._out_func(ins, self.layer_func, self._neuron_sel_and_head_map, neuron_selection, r_init)
        else:
            outs = self._out_func(ins, self.layer_func)

        return outs

    def try_apply(self, ins, callback=None, neuron_selection=None, stop_mapping_at_layers=None, r_init=None, f_init=None):
        """
        Tries to apply own forward pass:
        * Aggregates inputs and callbacks of all parent layers
        * Performs a canonization of the neuron_selection parameter
        * Calls wrap_hook (wrapped forward pass(es))
        * Calls _forward (forward result of forward pass to child layers)

        :param ins output of own forward pass
        :param neuron_selection: neuron_selection parameter. One of the following:
            - None or "all"
            - "max_activation"
            - int
            - list or np.array of int, with length equal to batch size
        :param stop_mapping_at_layers: None or list of layers to stop mapping at
        :param callback callback function of the parent layer that called self.try_apply
        :param r_init: None or Scalar or Array-Like or Dict {layer_name:scalar or array-like} reverse initialization value. Value with with explanation is initialized (i.e., head_mapping).
        :param f_init: None or Scalar or Array-Like or Dict {layer_name:scalar or array-like} forward initialization value. Value with which the forward is initialized.
        """
        # DEBUG
        #print(self.name, self.input_shape, np.shape(ins))

        if self.no_forward_pass == True and self.activations_saved == True:
            # calculate no forward pass, instead pass on to next layers

            self._forward(self.saved_forward_vals["outs"], neuron_selection, stop_mapping_at_layers, r_init, f_init)

            return

        self.reversed_output_vals = None

        #uses the class attribute, if it is not None.
        if self.r_init is not None:
            r_init = self.r_init

        if self.f_init is not None:
            f_init = self.f_init

        # aggregate inputs
        if self.input_vals is None:
            self.input_vals = []
        if self.activations_saved == False:
            # do not append same activation and callbacks already saved!
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

            # initialize explanation functions
            self.set_explain_functions(stop_mapping_at_layers)

            # set inputs to f_init, if it is not None
            if f_init is not None:
                if isinstance(f_init, dict):
                    if self.name in f_init.keys():
                        # f_init should be int or array-like. Shape should fit.
                        for i, in_val in enumerate(self.input_vals):
                            self.input_vals[i] = self._toNumber(in_val, f_init[self.name])
                else:
                    # f_init should be int or array-like. Shape should fit.
                    for i, in_val in enumerate(self.input_vals):
                        self.input_vals[i] = self._toNumber(in_val, f_init)

            # tensorify wrap_hook inputs as much as possible for graph efficiency
            input_vals = self.input_vals
            if len(input_vals) == 1:
                input_vals = input_vals[0]

            # adapt neuron_selection param
            if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
                if neuron_selection is None:
                    neuron_selection_tmp = None
                elif isinstance(neuron_selection, str) and neuron_selection == "all":
                    neuron_selection_tmp = None
                elif isinstance(neuron_selection, str) and neuron_selection == "max_activation":
                    neuron_selection_tmp = "max_activation"
                elif isinstance(neuron_selection, int) or isinstance(neuron_selection, np.int32):
                    neuron_selection_tmp = [[neuron_selection] for n in range(self.input_vals[0].shape[0])]
                    neuron_selection_tmp = tf.constant(neuron_selection_tmp)
                elif isinstance(neuron_selection, list) or (
                        hasattr(neuron_selection, "shape") and len(neuron_selection.shape) == 1):

                    neuron_selection_tmp = [[n] for n in neuron_selection]
                    neuron_selection_tmp = tf.constant(neuron_selection_tmp)
                else:
                    raise ValueError(
                        "Parameter neuron_selection only accepts the following values: None, 'all', 'max_activation', <int>, <list>, <one-dimensional array>")
            else:
                neuron_selection_tmp = neuron_selection

            # apply and wrappers
            if self.debug == True:
                print("forward hook", self.name)
            if self.saved_forward_vals is None:
                self.saved_forward_vals = {}
            self.saved_forward_vals["neuron_selection"] = neuron_selection_tmp
            self.saved_forward_vals["stop_mapping_at_layers"] = stop_mapping_at_layers
            self.saved_forward_vals["r_init"] = r_init
            self.saved_forward_vals["outs"] = self.compute_output(input_vals, neuron_selection_tmp, stop_mapping_at_layers, r_init)

            # forward
            self._forward(self.saved_forward_vals["outs"], neuron_selection, stop_mapping_at_layers, r_init, f_init)

    def set_explain_functions(self, stop_mapping_at_layers):
        self._explain_func = kfunctional.base_explanation

    def compute_explanation(self, ins, reversed_outs):
        """
        hook that computes the explanations.
        * Core XAI functionality

        :param ins: input(s) of this layer
        :param args: outputs of wrap_hook (any parameters that may be needed to compute explanation)
        :param reversed_outs: either backpropagated explanation(s) of child layers, or None if this is the last layer

        :returns explanation, or tensor of multiple explanations if the layer has multiple inputs (one for each)

        To be extended for specific XAI methods
        """
        #some preparation
        outs = self.saved_forward_vals["outs"]

        if reversed_outs is None:
            reversed_outs = outs

        #apply correct explanation function
        return self._explain_func(reversed_outs, len(self.input_shape), len(self.layer_next))


class GradientReplacementLayer(ReplacementLayer):
    """
    Simple extension of ReplacementLayer
    * Explains by computing gradients of outputs w.r.t. inputs of layer
    """
    def __init__(self, *args, **kwargs):
        super(GradientReplacementLayer, self).__init__(*args, **kwargs)

    def set_explain_functions(self, stop_mapping_at_layers):
        if len(self.layer_next) == 0 or (stop_mapping_at_layers is not None and self.name in stop_mapping_at_layers):
            self._explain_func = kfunctional.final_gradient_explanation
        else:
            self._explain_func = kfunctional.gradient_explanation

    def compute_explanation(self, ins, reversed_outs):
        # some preparation
        outs = self.saved_forward_vals["outs"]

        if reversed_outs is None:
            reversed_outs = outs

        # apply correct explanation function
        if len(self.layer_next) == 0 or (self.saved_forward_vals["stop_mapping_at_layers"] is not None and self.name in self.saved_forward_vals["stop_mapping_at_layers"]):
            ret = self._explain_func(ins,
                                     self.layer_func,
                                     self._neuron_sel_and_head_map,
                                     self._out_func,
                                     reversed_outs,
                                     len(self.input_shape),
                                     len(self.layer_next),
                                     self.saved_forward_vals["neuron_selection"],
                                     self.saved_forward_vals["r_init"],
                                     )
        else:
            ret = self._explain_func(ins, self.layer_func, self._out_func, reversed_outs, len(self.input_shape),
                                     len(self.layer_next))

        return ret


class ReverseModel():
    """
    Defines a ReverseModel

    ReverseModels are built from ReplacementLayer subclasses. A ReverseModel is defined via a list of Input ReplacementLayers (the input layers of the model)
            and ReplacementLayers (the whole model)

    Offers methods to
        - build
        - apply
        - get precomputed explanations from
        - get activations
        - save
        - load
    the ReverseModel
    """

    def __init__(self, model, reverse_mappings, default_reverse_mapping):
        self.build(model, reverse_mappings, default_reverse_mapping)

    def build(self, model, reverse_mappings, default_reverse_mapping):
        """
        Builds the ReverseModel by wrapping keras network layer(s) into ReplacementLayer(s)

        :param model: tf.keras model to be replaced
        :param reverse_mappings: mapping layer->reverse mapping (ReplacementLayer or some subclass thereof)
        :param default_reverse_mapping: ReplacementLayer or some subclass thereof; default mapping to use

        :returns -
        """

        # build model that is to be analyzed
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

        # connect graph structure
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

        # find input access points
        input_layers = []
        for i, t in enumerate(model.inputs):
            for layer in replacement_layers:
                if id(layer.layer_func.output) == id(t):
                    input_layers.append(layer)
            if len(input_layers) < i + 1:
                # if we did not append an input layer, we need to create one
                # TODO case for no input layer here
                raise ValueError("Temporary error. You need to explicitly define an Input Layer for now")

        self._reverse_model = (input_layers, replacement_layers)

    def apply(self, Xs, neuron_selection="max_activation", explained_layer_names=None, stop_mapping_at_layers=None, r_init=None, f_init=None):
        """
        Computes an explanation by applying the ReverseModel

        :param Xs: tensor or np.array of Input to be explained. Shape (n_ins, batch_size, ...) in model has multiple inputs, or (batch_size, ...) otherwise
        :param neuron_selection: neuron_selection parameter. Used to only compute explanation w.r.t. specific output neurons. One of the following:
                - None or "all"
                - "max_activation"
                - int
                - list or np.array of int, with length equal to batch size
        :param explained_layer_names: None or "all" or list of layer names whose explanations should be returned.
                                      Can be used to obtain intermediate explanations or explanations of multiple layers
        :param stop_mapping_at_layers: None or list of layers to stop mapping at ("output" layers)
        :param r_init: None or Scalar or Array-Like or Dict {layer_name:scalar or array-like} reverse initialization value. Value with which the explanation is initialized.
        :param f_init: None or Scalar or Array-Like or Dict {layer_name:scalar or array-like} forward initialization value. Value with which the forward is initialized.

        :returns Dict of the form {layer name (string): explanation (numpy.ndarray)}
        """
        # shape of Xs: (n_ins, batch_size, ...), or (batch_size, ...)

        reverse_ins, reverse_layers = self._reverse_model

        if stop_mapping_at_layers is not None and (isinstance(neuron_selection, int) or isinstance(neuron_selection, list) or isinstance(neuron_selection, np.ndarray)):
            warnings.warn("You are specifying layers to stop forward pass at, and also neuron-selecting by index. Please make sure the corresponding shapes fit together!")

        if not isinstance(Xs, tf.Tensor):
            try:
                Xs = tf.constant(Xs)
            except:
                raise ValueError("Xs has not supported type ", type(Xs))

        # format input & obtain explanations
        if len(reverse_ins) == 1:
            # single input network
            reverse_ins[0].try_apply(tf.constant(Xs), neuron_selection=neuron_selection,
                                     stop_mapping_at_layers=stop_mapping_at_layers, r_init=r_init, f_init=f_init)

        else:
            # multiple inputs. reshape to (n_ins, batch_size, ...)
            for i, reverse_in in enumerate(reverse_ins):
                reverse_in.try_apply(tf.constant(Xs[i]), neuron_selection=neuron_selection,
                                     stop_mapping_at_layers=stop_mapping_at_layers, r_init=r_init, f_init=f_init)

        # obtain explanations for specified layers
        hm = self.get_explanations(explained_layer_names)

        return hm


    def get_explanations(self, explained_layer_names=None):
        """
        Get results of (previously computed) explanation.
        explanation of layer i has shape equal to input_shape of layer i.

        :param explained_layer_names: None or "all" or list of strings containing the names of the layers.
                            if explained_layer_names == 'all', explanations of all layers are returned.
                            if None, return explanations of input layer only.

        :returns Dict of the form {layer name (string): explanation (numpy.ndarray)}

        """

        reverse_ins, reverse_layers = self._reverse_model

        hm = {}

        if explained_layer_names is None:
            # just explain input layers
            for layer in reverse_ins:
                hm[layer.name] = np.array(layer.explanation)

            return hm

        # output everything possible
        if explained_layer_names is "all":
            for layer in reverse_layers:
                if layer.explanation is not None:
                    hm[layer.name] = layer.explanation.numpy()

            return hm

        # otherwise, obtain explanations for specified layers
        for name in explained_layer_names:
            layer = [layer for layer in reverse_layers if layer.name == name]
            if len(layer) > 0:
                if layer[0].explanation is None:
                    raise AttributeError(f"layer <<{name}>> has to be analyzed before")
                hm[name] = layer[0].explanation.numpy()

        return hm

    def get_hook_activations(self, layer_names=None):

        """
        Get results of (previously computed) activations.
        activations of layer i has shape equal to output_shape of layer i.

        :param layer_names: None or list of strings containing the names of the layers.
                            if activations of last layer or layer after and inclusive stop_mapping_at are NOT available.
                            if None, return activations of input layer only.

        :returns Dict of the form {layer name (string): explanation (numpy.ndarray)}

        """

        reverse_ins, reverse_layers = self._reverse_model

        activations = {}

        if layer_names is None:
            # just explain input layers
            for layer in reverse_ins:
                if layer.activations_saved == False:
                    raise AttributeError("activations have to be saved first! Use for instance 'no_forward_pass=True'")
                activations[layer.name] = layer.saved_forward_vals

            return activations

        # output everything possible
        if layer_names is "all":
            for layer in reverse_layers:
                if layer.activations_saved == True:
                    activations[layer.name] = layer.saved_forward_vals

            return activations


        # otherwise, obtain explanations for specified layers
        for name in layer_names:
            layer = [layer for layer in reverse_layers if layer.name == name]
            if len(layer) > 0:
                if layer[0].activations_saved == False:
                    raise AttributeError(f"activations of <<{name}>> have to be saved first! Use for instance 'no_forward_pass=True'")
                activations[name] = layer[0].saved_forward_vals

        return activations


    #TODO
    def save(self):
        raise NotImplementedError

    #TODO
    def load(self):
        raise NotImplementedError