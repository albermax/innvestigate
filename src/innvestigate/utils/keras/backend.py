from __future__ import annotations

from typing import List

import tensorflow as tf
import tensorflow.keras.backend as kbackend

from innvestigate.utils import to_list
from innvestigate.utils.types import OptionalList, Tensor

# TODO: remove this file -A.

__all__ = [
    "gradients",
    "cast_to_floatx",
    "is_not_finite",
    "extract_conv2d_patches",
    "gather",
    "gather_nd",
]


def gradients(
    Xs: OptionalList[Tensor], Ys: OptionalList[Tensor], known_Ys: OptionalList[Tensor]
) -> List[Tensor]:
    grad = tf.gradients(Ys, Xs, grad_ys=known_Ys, stop_gradients=Xs)
    if grad is None:
        raise TypeError("Gradient computation failed, returned None.")
    return to_list(grad)


def is_not_finite(X: Tensor) -> Tensor:  # returns Tensor of dtype bool
    """Checks if tensor x is finite, if not throws an exception."""
    # x = tensorflow.check_numerics(x, "innvestigate - is_finite check")
    return tf.logical_not(tf.is_finite(X))


def cast_to_floatx(X: Tensor) -> Tensor:
    return tf.cast(X, dtype=kbackend.floatx())

def extract_conv2d_patches(X: Tensor, kernel_shape, strides, rates, padding) -> Tensor:
    """Extracts conv2d patches like TF function extract_image_patches.

    :param x: Input image.
    :param kernel_shape: Shape of the Keras conv2d kernel.
    :param strides: Strides of the Keras conv2d layer.
    :param rates: Dilation rates of the Keras conv2d layer.
    :param padding: Paddings of the Keras conv2d layer.
    :return: The extracted patches.
    """
    if kbackend.image_data_format() == "channels_first":
        X = kbackend.permute_dimensions(X, (0, 2, 3, 1))
    kernel_shape = [1, kernel_shape[0], kernel_shape[1], 1]
    strides = [1, strides[0], strides[1], 1]
    rates = [1, rates[0], rates[1], 1]
    ret = tf.extract_image_patches(X, kernel_shape, strides, rates, padding.upper())

    if kbackend.image_data_format() == "channels_first":
        # TODO: check if we need to permute again.xs
        pass
    return ret


def gather(
    X: Tensor,
    axis: Tensor,  # Tensor of integer dtype
    indices: Tensor,  # Tensor of integer dtype
) -> Tensor:
    """TensorFlow's gather:
    Gather slices from `X` axis `axis` according to `indices`.
    """
    return tf.gather(X, indices, axis=axis)


def gather_nd(
    X: Tensor,
    indices: Tensor,  # Tensor of integer dtype
) -> Tensor:
    """TensorFlow's gather_nd:
    Gather slices from `X` into a Tensor with shape specified by `indices`."""
    return tf.gather_nd(X, indices)
