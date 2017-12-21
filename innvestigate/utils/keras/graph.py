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


from ... import utils

import keras.engine.topology


__all__ = [
    "reverse_model",
]


###############################################################################
###############################################################################
###############################################################################


def reverse_model(model, reverse_mapping, default_reverse=None):

    reversed_tensors = {tmp:tmp for tmp in model.outputs}

    if not callable(reverse_mapping):
        reverse_mapping_data = reverse_mapping
        def reverse_mapping(layer):
            try:
                return reverse_mapping_data[type(layer)]
            except KeyError:
                return None

    def reverse_container(container, state):
        for layer_index, layer in list(enumerate(container.layers))[::-1]:
            if isinstance(layer, keras.engine.topology.Container):
                reverse_container(layer, state)
            else:
                # A layer can be shared, i.e., applied several times.
                # This leads to several in- and outbound tensors.
                for node_index in range(len(layer.inbound_nodes)):
                    Xs = utils.listify(layer.get_input_at(node_index))
                    Ys = utils.listify(layer.get_output_at(node_index))
                    reversed_Ys = [reversed_tensors[ys] for ys in Ys]

                    reverse_f = reverse_mapping(layer)
                    if reverse_f is None:
                        reverse_f = default_reverse
                    reversed_Xs = reverse_f(
                        Xs, Ys, reversed_Ys,
                        {
                            "reverse_id": state["reverse_id"],
                            "model": model,
                            "container": container,
                            "layer": layer,
                            "layer_index": layer_index,
                            "node_index": node_index,
                        })
                    reversed_Xs = utils.listify(reversed_Xs)
                    state["reverse_id"] += 1

                    for xs, reversed_xs in zip(Xs, reversed_Xs):
                        assert xs not in reversed_Ys
                        reversed_tensors[xs] = reversed_xs

    # [workaround for closures in python 2 and 3]
    # sequence id for reverse calls in our reversal routine
    state = {"reverse_id": 0}
    reverse_container(model, state)

    return [reversed_tensors[tmp] for tmp in model.inputs]
