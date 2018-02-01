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


from ... import utils as iutils

import keras.engine.topology


__all__ = [
    "contains_activation",
    "contains_kernel",
    "get_kernel",

    "get_layer_inbound_count",
    "get_layer_outbound_count",
    "get_layer_io",
    "get_layer_wo_activation",

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


def get_kernel(layer):
    ret = [x for x in layer.get_weights() if len(x.shape) > 1]
    assert len(ret) == 1
    return ret[0]


###############################################################################
###############################################################################
###############################################################################


def get_layer_inbound_count(layer):
    return len(layer.inbound_nodes)


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


def reverse_model(model, reverse_mapping,
                  default_reverse_mapping=None,
                  head_mapping=None,
                  verbose=False,
                  return_all_reversed_tensors=False):

    if head_mapping is None:
        def head_mapping(X):
            return X

    reversed_tensors = {tmp: {"id": (-1, i), "tensor": head_mapping(tmp)}
                        for i, tmp in enumerate(model.outputs)}

    if not callable(reverse_mapping):
        reverse_mapping_data = reverse_mapping

        def reverse_mapping(layer):
            try:
                return reverse_mapping_data[type(layer)]
            except KeyError:
                return None

    def _print(s):
        if verbose is True:
            print(s)
        pass

    def reverse_container(container, state):
        for layer_index, layer in list(enumerate(container.layers))[::-1]:
            if isinstance(layer, keras.engine.topology.Container):
                _print(" Reverse container: {}".format(layer))
                reverse_container(layer, state)
            else:
                # A layer can be shared, i.e., applied several times.
                # This leads to several in- and outbound tensors.
                for node_index in range(len(layer.inbound_nodes)):
                    Xs = iutils.listify(layer.get_input_at(node_index))
                    Ys = iutils.listify(layer.get_output_at(node_index))
                    reversed_Ys = [reversed_tensors[ys]["tensor"]
                                   for ys in Ys]
                    reverse_id = state["reverse_id"]
                    state["reverse_id"] += 1

                    _print("  [RID: {}] Reverse layer {}"
                           " -> node {} ({})".format(reverse_id, layer_index,
                                                     node_index, layer))
                    reverse_f = reverse_mapping(layer)
                    if reverse_f is None:
                        reverse_f = default_reverse_mapping
                    reversed_Xs = reverse_f(
                        Xs, Ys, reversed_Ys,
                        {
                            "reverse_id": reverse_id,
                            "model": model,
                            "container": container,
                            "layer": layer,
                            "layer_index": layer_index,
                            "node_index": node_index,
                        })
                    reversed_Xs = iutils.listify(reversed_Xs)

                    tmp = zip(Xs, reversed_Xs)
                    for i, (xs, reversed_xs) in enumerate(tmp):
                        assert xs not in reversed_Ys
                        reversed_tensors[xs] = {"id": (reverse_id, i),
                                                "tensor": reversed_xs}

    _print("Reverse model: {}".format(model))
    # [workaround for closures in python 2 and 3]
    # sequence id for reverse calls in our reversal routine
    state = {"reverse_id": 0}
    reverse_container(model, state)

    reversed_input_tensors = [reversed_tensors[tmp]["tensor"]
                              for tmp in model.inputs]
    if return_all_reversed_tensors is True:
        return reversed_input_tensors, reversed_tensors
    else:
        return reversed_input_tensors
