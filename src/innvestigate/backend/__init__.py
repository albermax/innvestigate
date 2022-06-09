from __future__ import annotations

from typing import TypeVar

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kbackend

from innvestigate.backend.types import Layer, OptionalList, ShapeTuple, Tensor

__all__ = [
    "to_list",
    "unpack_singleton",
    "shape",
    "batch_size",
    "gradients",
    "cast_to_floatx",
    "is_not_finite",
    "safe_divide",
    "count_non_zero",
    "add_gaussian_noise",
    "apply_mask",
    "apply",
    "broadcast_np_tensors_to_keras_tensors",
    "extract_conv2d_patches",
]

_EPS = kbackend.epsilon()

T = TypeVar("T")  # Generic type, can be anything


def disable_eager_execution() -> None:
    tf.compat.v1.disable_eager_execution()


def to_list(X: OptionalList[T]) -> list[T]:
    """Wraps tensor `X` into a list, if it isn't a list of Tensors yet."""
    if isinstance(X, list):
        return X
    return [X]


def unpack_singleton(x: OptionalList[T]) -> OptionalList[T]:
    """Gets the first element of a list if it has only one value.

    Otherwise return the list.

    # Argument
        x: A list or singleton.

    # Returns
        The same list or the first element.
    """
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x


def shape(X: Tensor) -> list[int | None]:
    """Return shape of Tensor as list of ints."""
    shape: list[int | None] = X.get_shape().as_list()
    return shape


def batch_size(X: Tensor) -> int:
    """Return batch size of Tensor as integer."""
    bs: int | None = X.get_shape()[0]
    if isinstance(bs, int):
        return bs
    raise ValueError(f"Found non-integer batch_size {bs} for Tensor {X}")


def gradients(
    Xs: OptionalList[Tensor], Ys: OptionalList[Tensor], known_Ys: OptionalList[Tensor]
) -> list[Tensor]:
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
    return tf.logical_not(tf.math.is_finite(X))


def cast_to_floatx(X: Tensor) -> Tensor:
    return tf.cast(X, kbackend.floatx())


def safe_divide(A: Tensor, B: Tensor, factor: float = _EPS) -> Tensor:
    """Divide A by B, replacing all zeroes in B with `factor`."""
    is_zero = cast_to_floatx(kbackend.equal(B, kbackend.constant(0)))
    return A / (B + factor * is_zero)


def count_non_zero(X: Tensor, axis, keepdims: bool = False) -> Tensor:
    "Count non-zero elements in tensor."
    non_zeros = cast_to_floatx(kbackend.not_equal(X, kbackend.constant(0)))
    return kbackend.sum(non_zeros, axis=axis, keepdims=keepdims)


def add_gaussian_noise(X: Tensor, mean: float = 0.0, stddev: float = 1.0) -> Tensor:
    """Add Gaussian noise to tensor `X`."""
    return X + kbackend.random_normal(
        shape=tf.shape(X), mean=mean, stddev=stddev, dtype=X.dtype
    )


def apply_mask(Xs: list[T], mask: list[bool]) -> list[T]:
    """Apply mask to list `Xs`, keeping only the elements for which
    mask is True.

    :param Xs: List to be masked.
    :type Xs: List[T]
    :param mask: Mask applied to `Xs`. If True, corresponding entry in `Xs` is kept.
    :type mask: List[bool]
    :return: Masked list.
    :rtype: List[T]
    """
    if len(Xs) != len(mask):
        raise ValueError("mask not of same length as list that is to be masked.")
    return [x for x, keep in zip(Xs, mask) if keep]


###############################################################################


def apply(layer: Layer, inputs: OptionalList[Tensor]) -> list[Tensor]:
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

    return to_list(ret)


def broadcast_np_tensors_to_keras_tensors(
    np_tensors: float | np.ndarray | list[np.ndarray],
    keras_tensors: OptionalList[Tensor],
) -> list[np.ndarray]:
    """Broadcasts numpy tensors to the shape of Keras tensors.

    :param np_tensors: Numpy tensors that should be broadcasted.
    :type np_tensors: Union[np.ndarray, List[np.ndarray]]
    :param keras_tensors: The Keras tensors with the target shapes.
    :type keras_tensors: OptionalList[Tensor]
    :return: The broadcasted Numpy tensors.
    :rtype: List[np.ndarray]
    """

    def none_to_one(shape: ShapeTuple):
        return [1 if dim is None else dim for dim in shape]

    keras_tensors = to_list(keras_tensors)

    if isinstance(np_tensors, list):
        return [
            np.broadcast_to(n, none_to_one(kbackend.int_shape(k)))
            for k, n in zip(keras_tensors, np_tensors)
        ]
    return [
        np.broadcast_to(np_tensors, none_to_one(kbackend.int_shape(k)))
        for k in keras_tensors
    ]


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
    ret = tf.compat.v1.extract_image_patches(
        X, kernel_shape, strides, rates, padding.upper()
    )

    if kbackend.image_data_format() == "channels_first":
        # TODO: check if we need to permute again.xs
        pass
    return ret
