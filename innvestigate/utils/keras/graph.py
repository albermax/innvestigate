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


import inspect
import keras.backend as K
import keras.engine.topology
import keras.layers
import keras.models
import numpy as np


from .. import keras as kutils
from ... import layers as ilayers
from ... import utils as iutils


__all__ = [
    "contains_activation",
    "contains_kernel",
    "is_container",
    "is_convnet_layer",
    "is_relu_convnet_layer",

    "get_kernel",

    "get_layer_inbound_count",
    "get_layer_outbound_count",
    "get_layer_neuronwise_io",
    "copy_layer_wo_activation",
    "copy_layer",
    "pre_softmax_tensors",
    "model_wo_softmax",

    "get_model_layers",
    "model_contains",

    "trace_model_execution",

    "ReverseMappingBase",
    "reverse_model",
]


###############################################################################
###############################################################################
###############################################################################


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "activation"):
        if activation is not None:
            return layer.activation == keras.activations.get(activation)
        else:
            return True
    else:
        return False


def contains_kernel(layer):
    """
    Check whether the layer contains a kernel.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "kernel"):
        return True
    else:
        return False


def contains_bias(layer):
    """
    Check whether the layer contains a bias.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "bias"):
        return True
    else:
        return False


def is_container(layer):
    return isinstance(layer, keras.engine.topology.Container)


def is_convnet_layer(layer):
    # todo: add checks, e.g., no recurrent layers
    return True


def is_relu_convnet_layer(layer):
    return (is_convnet_layer(layer) and
            (not contains_activation(layer) or
             contains_activation(layer, None) or
             contains_activation(layer, "linear") or
             contains_activation(layer, "relu")))


def is_input_layer(layer):
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object or
    # the layer itself is the first layer.

    layer_inputs = get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        keras.layers.Flatten,
        keras.layers.Permute,
        keras.layers.Reshape,
    )
    while any([isinstance(x, IGNORED_LAYERS) for x in layer_inputs]):
        tmp = set()
        for l in layer_inputs:
            if isinstance(l, IGNORED_LAYERS):
                tmp.update(get_input_layers(l))
            else:
                tmp.add(l)
        layer_inputs = tmp

    if all([isinstance(x, keras.layers.InputLayer)
            for x in layer_inputs]):
        return True
    elif getattr(layer, "input_shape", None) is not None:
        # relies on Keras convention
        return True
    elif getattr(layer, "batch_input_shape", None) is not None:
        # relies on Keras convention
        return True
    else:
        return False


###############################################################################
###############################################################################
###############################################################################


def get_kernel(layer):
    ret = [x for x in layer.get_weights() if len(x.shape) > 1]
    assert len(ret) == 1
    return ret[0]


def get_input_layers(layer):
    ret = set()

    for node_index in range(len(layer._inbound_nodes)):
        Xs = iutils.to_list(layer.get_input_at(node_index))
        for X in Xs:
            ret.add(X._keras_history[0])

    return ret


###############################################################################
###############################################################################
###############################################################################


def get_layer_inbound_count(layer):
    return len(layer._inbound_nodes)


def get_layer_outbound_count(layer):
    return len(layer.outbound_nodes)


def get_layer_neuronwise_io(layer, node_index=0, return_i=True, return_o=True):
    if not contains_kernel(layer):
        raise NotImplementedError()

    Xs = iutils.to_list(layer.get_input_at(node_index))
    Ys = iutils.to_list(layer.get_output_at(node_index))

    if isinstance(layer, keras.layers.Dense):
        # Xs and Ys are already in shape.
        ret_Xs = Xs
        ret_Ys = Ys
    elif isinstance(layer, keras.layers.Conv2D):
        kernel = get_kernel(layer)
        # Expect filter dimension to be last.
        n_channels = kernel.shape[-1]

        if return_i:
            extract_patches = ilayers.ExtractConv2DPatches(kernel.shape[:2],
                                                           kernel.shape[2],
                                                           layer.strides,
                                                           layer.dilation_rate,
                                                           layer.padding)
            # shape [samples, out_row, out_col, weight_size]
            reshape = ilayers.Reshape((-1, np.product(kernel.shape[:3])))
            ret_Xs = [reshape(extract_patches(x)) for x in Xs]

        if return_o:
            # Get Ys into shape (samples, channels)
            if K.image_data_format() == "channels_first":
                # Ys shape is [samples, channels, out_row, out_col]
                def reshape(x):
                    x = ilayers.Transpose((0, 2, 3, 1))(x)
                    x = ilayers.Reshape((-1, n_channels))(x)
                    return x
            else:
                # Ys shape is [samples, out_row, out_col, channels]
                def reshape(x):
                    x = ilayers.Reshape((-1, n_channels))(x)
                    return x
            ret_Ys = [reshape(x) for x in Ys]

    else:
        raise NotImplementedError()

    # Xs is (n, d) and Ys is (d, channels)
    if return_i and return_o:
        return ret_Xs, ret_Ys
    elif return_i:
        return ret_Xs
    elif return_o:
        return ret_Ys
    else:
        raise Exception()


def get_layer_from_config(old_layer, new_config, weights=None):
    new_layer = old_layer.__class__.from_config(new_config)

    if weights is None:
        weights = old_layer.get_weights()

    if len(weights) > 0:
        # init weights
        n_nodes = get_layer_inbound_count(old_layer)
        for i in range(n_nodes):
            new_layer(old_layer.get_input_at(i))
        new_layer.set_weights(weights)

    return new_layer


def copy_layer_wo_activation(layer,
                             keep_bias=True,
                             name_template=None,
                             weights=None):
    config = layer.get_config()
    if name_template is None:
        config["name"] = None
    else:
        config["name"] = name_template % config["name"]
    if contains_activation(layer):
        config["activation"] = None
    if keep_bias is False and config.get("use_bias", False):
        config["use_bias"] = False
        if weights is None:
            weights = layer.get_weights()[:-1]
    return get_layer_from_config(layer, config, weights=weights)


def copy_layer(layer,
               keep_bias=True,
               name_template=None,
               weights=None):

    config = layer.get_config()
    if name_template is None:
        config["name"] = None
    else:
        config["name"] = name_template % config["name"]
    if keep_bias is False and config.get("use_bias", False):
        config["use_bias"] = False
        if weights is None:
            weights = layer.get_weights()[:-1]
    return get_layer_from_config(layer, config, weights=weights)


def pre_softmax_tensors(Xs, should_find_softmax=True):
    softmax_found = False

    Xs = iutils.to_list(Xs)
    ret = []
    for x in Xs:
        layer, node_index, tensor_index = x._keras_history
        if contains_activation(layer, activation="softmax"):
            softmax_found = True
            if isinstance(layer, keras.layers.Activation):
                ret.append(layer.get_input_at(node_index)[0])
            else:
                layer_wo_act = copy_layer_wo_activation(layer)
                ret.append(layer_wo_act(layer.get_input_at(node_index)))

    if should_find_softmax and not softmax_found:
        raise Exception("No softmax found.")

    return ret


def model_wo_softmax(model):
    return keras.model.Model(inputs=model.inputs,
                             outputs=pre_softmax_tensor(model.outputs),
                             name=model.name)


###############################################################################
###############################################################################
###############################################################################


def get_model_layers(model):
    ret = []

    def collect_layers(container):
        for layer in container.layers:
            assert layer not in ret
            ret.append(layer)
            if is_container(layer):
                collect_layers(layer)
    collect_layers(model)

    return ret


def model_contains(model, layer_condition, return_only_counts=False):
    if callable(layer_condition):
        layer_condition = [layer_condition, ]
        single_condition = True
    else:
        single_condition = False

    layers = get_model_layers(model)
    collected_layers = []
    for condition in layer_condition:
        tmp = [layer for layer in layers if condition(layer)]
        collected_layers.append(tmp)

    if return_only_counts is True:
        collected_layers = [len(v) for v in collected_layers]

    if single_condition is True:
        return collected_layers[0]
    else:
        return collected_layers


###############################################################################
###############################################################################
###############################################################################


def trace_model_execution(model, reapply_on_copied_layers=False):

    # Get all layers in model.
    layers = get_model_layers(model)

    # Check if some layers are containers.
    # Ignoring the outermost container, i.e. the passed model.
    contains_container = any([((l is not model) and is_container(l))
                              for l in layers])

    # If so rebuild the graph, otherwise recycle computations,
    # and create executed node list. (Keep track of paths?)
    if contains_container is True:
        # When containers/models are used as layers, then layers
        # inside the container/model do not keep track of nodes.
        # This makes it impossible to iterate of the nodes list and
        # recover the input output tensors. (see else clause)
        #
        # To recover the computational graph we need to re-apply it.
        # This implies that the tensors-object we use for the forward
        # pass are different to the passed model. This it not the case
        # for the else clause.
        #
        # Note that reapplying the model does only change the inbound
        # and outbound nodes of the model itself. We copy the model
        # so the passed model should not be affected from the
        # reapplication.
        executed_nodes = []

        # Monkeypatch the call function in all the used layer classes.
        monkey_patches = [(layer, getattr(layer, "call")) for layer in layers]
        try:
            def patch(self, method):
                if hasattr(method, "__patched__") is True:
                    raise Exception("Should not happen as we patch "
                                    "objects not classes.")

                def f(*args, **kwargs):
                    input_tensors = args[0]
                    output_tensors = method(*args, **kwargs)
                    executed_nodes.append((self,
                                           input_tensors,
                                           output_tensors))
                    return output_tensors
                f.__patched__ = True
                return f

            # Apply the patches.
            for layer in layers:
                setattr(layer, "call", patch(layer, getattr(layer, "call")))

            # Trigger reapplication of model.
            model_copy = keras.models.Model(inputs=model.inputs,
                                            outputs=model.outputs)
            outputs = iutils.to_list(model_copy(model.inputs))
        finally:
            # Revert the monkey patches
            for layer, old_method in monkey_patches:
                setattr(layer, "call", old_method)

        # Now we have the problem that all the tensors
        # do not have a keras_history attribute as they are not part
        # of any node. Apply the flat model to get it.
        from . import apply as kapply
        new_executed_nodes = []
        tensor_mapping = {tmp: tmp for tmp in model.inputs}
        if reapply_on_copied_layers is True:
            layer_mapping = {layer: copy_layer(layer) for layer in layers}
        else:
            layer_mapping = {layer: layer for layer in layers}

        for layer, Xs, Ys in executed_nodes:
            layer = layer_mapping[layer]
            Xs, Ys = iutils.to_list(Xs), iutils.to_list(Ys)

            if isinstance(layer, keras.layers.InputLayer):
                # Special case. Do nothing.
                new_Xs, new_Ys = Xs, Ys
            else:
                new_Xs = [tensor_mapping[x] for x in Xs]
                new_Ys = iutils.to_list(kapply(layer, new_Xs))

            tensor_mapping.update({k: v for k, v in zip(Ys, new_Ys)})
            new_executed_nodes.append((layer, new_Xs, new_Ys))

        layers = [layer_mapping[layer] for layer in layers]
        outputs = [tensor_mapping[x] for x in outputs]
        executed_nodes = new_executed_nodes
    else:
        # Easy and safe way.
        reverse_executed_nodes = [
            (node.outbound_layer, node.input_tensors, node.output_tensors)
            for depth in sorted(model._nodes_by_depth.keys())
            for node in model._nodes_by_depth[depth]
        ]
        outputs = model.outputs

        executed_nodes = reversed(reverse_executed_nodes)
        pass

    # This list contains potentially nodes that are not part
    # final execution graph.
    # E.g., a layer was also applied outside of the model. Then its
    # node list contains nodes that do not contribute to the model's output.
    # Those nodes are filtered here.
    used_as_input = [x for x in outputs]
    tmp = []
    for l, Xs, Ys in reversed(list(executed_nodes)):
        if all([y in used_as_input for y in Ys]):
            used_as_input += Xs
            tmp.append((l, Xs, Ys))
    executed_nodes = list(reversed(tmp))

    return layers, executed_nodes, outputs


###############################################################################
###############################################################################
###############################################################################


class ReverseMappingBase(object):

    def __init__(self, layer, state):
        pass

    def apply(self, Xs, Yx, reversed_Ys, reverse_state):
        raise NotImplementedError()


def reverse_model(model, reverse_mappings,
                  default_reverse_mapping=None,
                  head_mapping=None,
                  verbose=False,
                  return_all_reversed_tensors=False,
                  clip_all_reversed_tensors=False,
                  project_bottleneck_tensors=False,
                  reapply_on_copied_layers=False):

    # Set default values ######################################################

    if head_mapping is None:
        def head_mapping(X):
            return X

    if not callable(reverse_mappings):
        # not callable, assume a dict that maps from layer to mapping
        reverse_mapping_data = reverse_mappings

        def reverse_mappings(layer):
            try:
                return reverse_mapping_data[type(layer)]
            except KeyError:
                return None

    def _print(s):
        if verbose is True:
            print(s)
        pass

    # Initialize structure that keeps track of reversed tensors ###############

    reversed_tensors = {}

    def add_reversed_tensors(reverse_id,
                             tensors_list,
                             reversed_tensors_list):

        def add_reversed_tensor(i, X, reversed_X):
            if project_bottleneck_tensors is not False:
                # todo: add check if is bottleneck tensor.
                if True:
                    project = ilayers.Project(*project_bottleneck_tensors)
                    reversed_X = project(X)

            if clip_all_reversed_tensors is not False:
                clip = ilayers.Clip(*clip_all_reversed_tensors)
                reversed_X = clip(reversed_X)

            if X not in reversed_tensors:
                reversed_tensors[X] = {"id": (reverse_id, i),
                                        "tensor": reversed_X}
            else:
                tmp = reversed_tensors[X]
                if "tensor" in tmp and "tensors" in tmp:
                    raise Exception("Wrong order, tensors already aggregated!")
                if "tensor" in tmp:
                    tmp["tensors"] = [tmp["tensor"], reversed_X]
                    del tmp["tensor"]
                else:
                    tmp["tensors"].append(reversed_X)

        tmp = zip(tensors_list, reversed_tensors_list)
        for i, (X, reversed_X) in enumerate(tmp):
            add_reversed_tensor(i, X, reversed_X)

    def get_reversed_tensor(tensor):
        tmp = reversed_tensors[tensor]
        if "tensor" not in tmp:
            tmp["tensor"] = keras.layers.Add()(tmp["tensors"])
        return tmp["tensor"]

    # Reverse the model #######################################################
    _print("Reverse model: {}".format(model))

    # Create a list with nodes in reverse execution order.
    layers, execution_list, outputs = trace_model_execution(
        model,
        reapply_on_copied_layers=reapply_on_copied_layers)
    reverse_execution_list = reversed(execution_list)

    # Initialize the reverse mapping functions.
    initialized_reverse_mappings = {}
    for layer in layers:
        # A layer can be shared, i.e., applied several times.
        # Allow to share a ReverMappingBase for each layer instance
        # in order to reduce the overhead.

        reverse_mapping = reverse_mappings(layer)
        if reverse_mapping is None:
            reverse_mapping = default_reverse_mapping

        if(inspect.isclass(reverse_mapping) and
           issubclass(reverse_mapping, ReverseMappingBase)):
            reverse_mapping_obj = reverse_mapping(
                layer,
                {
                    "model": model,
                    "layer": layer,
                }
            )
            reverse_mapping = reverse_mapping_obj.apply

        initialized_reverse_mappings[layer] = reverse_mapping

    # Initialize the reverse tensor mappings.
    add_reversed_tensors(-1,
                         outputs,
                         [head_mapping(tmp) for tmp in outputs])

    # Follow the list and revert the graph.
    for reverse_id, (layer, Xs, Ys) in enumerate(reverse_execution_list):
        if isinstance(layer, keras.layers.InputLayer):
            # Special case. Do nothing.
            pass
        elif is_container(layer):
            raise Exception("This is not supposed to happen!")
        else:
            Xs, Ys = iutils.to_list(Xs), iutils.to_list(Ys)
            if not all([ys in reversed_tensors for ys in Ys]):
                # This node is not part of our computational graph.
                # The (node-)world is bigger than this model.
                continue
            reversed_Ys = [get_reversed_tensor(ys)
                           for ys in Ys]

            _print("  [RID: {}] Reverse layer {}".format(reverse_id, layer))
            reverse_mapping = initialized_reverse_mappings[layer]
            reversed_Xs = reverse_mapping(
                Xs, Ys, reversed_Ys,
                {
                    "reverse_id": reverse_id,
                    "model": model,
                    "layer": layer,
                })
            reversed_Xs = iutils.to_list(reversed_Xs)
            add_reversed_tensors(reverse_id, Xs, reversed_Xs)

    # Return requested values #################################################
    reversed_input_tensors = [get_reversed_tensor(tmp)
                              for tmp in model.inputs]
    if return_all_reversed_tensors is True:
        return reversed_input_tensors, reversed_tensors
    else:
        return reversed_input_tensors
