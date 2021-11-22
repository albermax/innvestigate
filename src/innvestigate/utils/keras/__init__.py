from __future__ import annotations

from builtins import zip
from typing import List, Union

import numpy as np
import tensorflow.keras.backend as kbackend

import innvestigate.utils as iutils
from innvestigate.utils.types import Layer, OptionalList, Tensor

__all__ = [
    "apply",
    "broadcast_np_tensors_to_keras_tensors",
]


def apply(layer: Layer, inputs: OptionalList[Tensor]) -> List[Tensor]:
    """
    Apply a layer to input[s].

    A flexible apply that tries to fit input to layers expected input.
    This is useful when one doesn't know if a layer expects a single tensor
    or many.

    :param layer: A Keras layer instance.
    :type layer: Layer
    :param inputs: A list of input tensors or a single tensor.
    :type inputs: OptionalList[Tensor]
    :return: Output from applying the layer to the input.
    :rtype: List[Tensor]
    """

    if isinstance(inputs, list) and len(inputs) > 1:
        try:
            ret = layer(inputs)
        except (TypeError, AttributeError) as err:
            # layer expects a single tensor.
            if len(inputs) != 1:
                raise ValueError("Layer expects only a single input!") from err
            ret = layer(inputs[0])
    else:
        ret = layer(inputs[0])

    return iutils.to_list(ret)


def broadcast_np_tensors_to_keras_tensors(
    keras_tensors: OptionalList[Tensor],
    np_tensors: Union[float, np.ndarray, List[np.ndarray]],
) -> List[np.ndarray]:
    """Broadcasts numpy tensors to the shape of Keras tensors.

    :param keras_tensors: The Keras tensors with the target shapes.
    :type keras_tensors: OptionalList[Tensor]
    :param np_tensors: Numpy tensors that should be broadcasted.
    :type np_tensors: Union[np.ndarray, List[np.ndarray]]
    :return: The broadcasted Numpy tensors.
    :rtype: List[np.ndarray]
    """

    def none_to_one(tmp):
        return [1 if x is None else x for x in tmp]

    keras_tensors = iutils.to_list(keras_tensors)

    if isinstance(np_tensors, list):
        return [
            np.broadcast_to(ri, none_to_one(kbackend.int_shape(x)))
            for x, ri in zip(keras_tensors, np_tensors)
        ]
    return [
        np.broadcast_to(np_tensors, none_to_one(kbackend.int_shape(x)))
        for x in keras_tensors
    ]
