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
import keras.engine.topology
import keras.layers


from ... import utils as iutils


__all__ = [
    "contains_activation",
    "contains_kernel",
    "is_convnet_layer",
    "is_relu_convnet_layer",

    "get_kernel",

    "get_layer_inbound_count",
    "get_layer_outbound_count",
    "get_layer_io",
    "get_layer_wo_activation",

    "model_contains",

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
        Xs = iutils.listify(layer.get_input_at(node_index))
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


def get_layer_io(layer, node_index=0):
    Xs = iutils.listify(layer.get_input_at(node_index))
    Ys = iutils.listify(layer.get_output_at(node_index))
    return Xs, Ys


def get_layer_from_config(old_layer, new_config, weights=None):
    new_layer = old_layer.__class__.from_config(new_config)

    if weights is None:
        weights = old_layer.get_weights()

    if len(weights) > 0:
        # init weights
        new_layer(old_layer.get_input_at(0))
        new_layer.set_weights(weights)

    return new_layer


def get_layer_wo_activation(layer,
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


###############################################################################
###############################################################################
###############################################################################


def get_model_layers(model):
    ret = []

    def collect_layers(container):
        for layer in container.layers:
            assert layer not in ret
            ret.append(layer)
            if isinstance(layer, keras.engine.topology.Container):
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


class ReverseMappingBase(object):

    def __init__(self, layer, state):
        pass

    def apply(self, Xs, Yx, reversed_Ys, reverse_state):
        raise NotImplementedError()


def reverse_model(model, reverse_mappings,
                  default_reverse_mapping=None,
                  head_mapping=None,
                  verbose=False,
                  return_all_reversed_tensors=False):
    # In principle one can follow the graph of a keras model by mimicking
    # the BF-traversal as in keras.engine.topology:Container:run_internal_graph
    # There are two problem with this:
    # * [Minor] We rebuild the graph and it will be disjoint from the input
    #           models graph.
    # * [Major] Given nested models, we actually cannot track back to the
    #           original tensors as their nodes are not stored in the
    #           aforementioned function. Therefore in the case of nested
    #           models we need to rebuild the graph to invert it.
    # * [Major] To do the inversion properly we need to follow the nodes by
    #           depth and cannot go layer by layer. On the other hand we
    #           would like to support the ReverseMappingBase interface that
    #           allows to cache the reversion.

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

        def add_reversed_tensor(i, xs, reversed_xs):
            assert xs not in reversed_tensors
            reversed_tensors[xs] = {"id": (reverse_id, i),
                                    "tensor": reversed_xs}

        tmp = zip(tensors_list, reversed_tensors_list)
        for i, (xs, reversed_xs) in enumerate(tmp):
            add_reversed_tensor(i, xs, reversed_xs)

    # Reverse the model #######################################################
    _print("Reverse model: {}".format(model))

    # Get all layers in model.
    layers = get_model_layers(model)

    # Check if some layers are containers.
    # Ignoring the outermost container, i.e. the passed model.
    contains_container = any([((l is not model) and
                               isinstance(l, keras.engine.topology.Container))
                              for l in layers])

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

    # If so rebuild the graph, otherwise recycle computations,
    # and create node execution list. (Keep track of paths?)
    if contains_container is True:
        raise NotImplementedError()
        pass
    else:
        # Easy and safe way.
        reverse_execution_list = [
            (node.outbound_layer, node.input_tensors, node.output_tensors)
            for depth in sorted(model._nodes_by_depth.keys())
            for node in model._nodes_by_depth[depth]
        ]
        inputs = model.inputs
        outputs = model.outputs
        pass

    # Initialize the reverse tensor mappings.
    add_reversed_tensors(-1,
                         outputs,
                         [head_mapping(tmp) for tmp in outputs])

    # Follow the list in reverse order and revert the graph.
    for reverse_id, (layer, Xs, Ys) in enumerate(reverse_execution_list):
        if isinstance(layer, keras.layers.InputLayer):
            # Special case. Do nothing.
            pass
        elif isinstance(layer, keras.engine.topology.Container):
            raise Exception("This is not supposed to happen!")
        else:
            Xs, Ys = iutils.listify(Xs), iutils.listify(Ys)

            if not all([ys in reversed_tensors for ys in Ys]):
                # This node is not part of our computational graph.
                # The (node-)world is bigger than this model.
                continue
            reversed_Ys = [reversed_tensors[ys]["tensor"]
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
            reversed_Xs = iutils.listify(reversed_Xs)

            add_reversed_tensors(reverse_id, Xs, reversed_Xs)

    # Return requested values #################################################
    reversed_input_tensors = [reversed_tensors[tmp]["tensor"]
                              for tmp in inputs]
    if return_all_reversed_tensors is True:
        return reversed_input_tensors, reversed_tensors
    else:
        return reversed_input_tensors
