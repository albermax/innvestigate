"""Utilities for the Keras backend."""
from __future__ import annotations

from typing import List

import tensorflow as tf
import tensorflow.keras.backend as kbackend
from tensorflow.python.ops import array_ops

from innvestigate.utils import to_list
from innvestigate.utils.types import OptionalList, Tensor

# TODO: remove this file -A.

__all__ = [
    "gradients",
    "cast_to_floatx",
    "is_not_finite",
    "safe_divide",
    "count_non_zero",
    "add_gaussian_noise",
    "extract_conv2d_patches",
]

_EPS = kbackend.epsilon()


def gradients(
    Xs: OptionalList[Tensor], Ys: OptionalList[Tensor], known_Ys: OptionalList[Tensor]
) -> List[Tensor]:
    if len(Ys) != len(known_Ys):
        raise ValueError(
            "Gradient computation failesd, Ys and known_Ys not of same length"
        )

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


def safe_divide(A: Tensor, B: Tensor, factor: float = _EPS) -> Tensor:
    """Divide A by B, replacing all zeroes in B with `factor`."""
    is_zero = cast_to_floatx(kbackend.equal(B, kbackend.constant(0)))
    return A / (B + factor * is_zero)


def count_non_zero(X: Tensor, axis, keepdims: bool) -> Tensor:
    "Count non-zero elements in tensor."
    non_zeros = cast_to_floatx(kbackend.not_equal(X, kbackend.constant(0)))
    return kbackend.sum(non_zeros, axis=axis, keepdims=keepdims)


def add_gaussian_noise(X: Tensor, mean: float = 0.0, stddev: float = 1.0) -> Tensor:
    """Add Gaussian noise to tensor `X`."""
    return X + kbackend.random_normal(
        shape=array_ops.shape(X), mean=mean, stddev=stddev, dtype=X.dtype
    )


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
