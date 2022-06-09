from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers

import innvestigate.backend as ibackend
from innvestigate.backend.types import OptionalList, ShapeTuple, Tensor

__all__ = [
    "OnesLike",
    "AsFloatX",
    "FiniteCheck",
    "GreaterThanZero",
    "LessEqualThanZero",
    "Sum",
    "Identity",
    "Abs",
    "Square",
    "Clip",
    "Project",
    "SafeDivide",
    "Repeat",
    "ReduceMean",
    "Reshape",
    "AugmentationToBatchAxis",
    "AugmentationFromBatchAxis",
    "MultiplyWithLinspace",
    "AddGaussianNoise",
    "ExtractConv2DPatches",
    "RunningMeans",
    "Broadcast",
    "MaxNeuronSelection",
    "MaxNeuronIndex",
    "NeuronSelection",
]


###############################################################################


class OnesLike(klayers.Layer):
    """Create list of all-ones tensors of the same shapes as provided tensors."""

    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> list[Tensor]:
        return [kbackend.ones_like(x) for x in ibackend.to_list(inputs)]


class AsFloatX(klayers.Layer):
    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> list[Tensor]:
        return [ibackend.cast_to_floatx(x) for x in ibackend.to_list(inputs)]


class FiniteCheck(klayers.Layer):
    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> list[Tensor]:
        return [
            kbackend.sum(ibackend.cast_to_floatx(ibackend.is_not_finite(x)))
            for x in ibackend.to_list(inputs)
        ]


###############################################################################


class _Reduce(klayers.Layer):
    def __init__(
        self,
        *args,
        axis: OptionalList[int] | None = -1,
        keepdims: bool = False,
        **kwargs,
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(*args, **kwargs)

    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> Tensor:
        return self._apply_reduce(inputs, axis=self.axis, keepdims=self.keepdims)

    def _apply_reduce(
        self,
        inputs: OptionalList[Tensor],
        axis: OptionalList[int] | None,
        keepdims: bool,
    ) -> Tensor:
        raise NotImplementedError()


class Sum(_Reduce):
    def _apply_reduce(
        self,
        inputs: OptionalList[Tensor],
        axis: OptionalList[int] | None,
        keepdims: bool,
    ) -> Tensor:
        return kbackend.sum(inputs, axis=axis, keepdims=keepdims)


###############################################################################


class _Map(klayers.Layer):
    def call(
        self, inputs: OptionalList[Tensor], *_args, **_kwargs
    ) -> OptionalList[Tensor]:
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        return self._apply_map(inputs)

    def _apply_map(self, X: Tensor):
        raise NotImplementedError()


class Identity(_Map):
    def _apply_map(self, X: Tensor) -> Tensor:
        return tf.identity(X)


class Abs(_Map):
    def _apply_map(self, X: Tensor) -> Tensor:
        return kbackend.abs(X)


class Square(_Map):
    def _apply_map(self, X: Tensor) -> Tensor:
        return kbackend.square(X)


class Clip(_Map):
    def __init__(
        self, min_value: float | int | Tensor, max_value: float | int | Tensor
    ) -> None:
        self._min_value = min_value
        self._max_value = max_value
        super().__init__()

    def _apply_map(self, X: Tensor) -> Tensor:
        return kbackend.clip(X, self._min_value, self._max_value)


class Project(_Map):
    def __init__(self, output_range=False, input_is_postive_only: bool = False) -> None:
        # TODO: add type of output_range
        self._output_range = output_range
        self._input_is_positive_only = input_is_postive_only
        super().__init__()

    def _apply_map(self, X: Tensor):
        dims: tuple[int] = kbackend.int_shape(X)
        n_dim: int = len(dims)
        axes = tuple(range(1, n_dim))

        if len(axes) == 1:
            # TODO(albermax): this is only the case when the dimension in this
            # axis is 1, fix this.
            # Cannot reduce
            return X

        absmax = kbackend.max(kbackend.abs(X), axis=axes, keepdims=True)
        X = ibackend.safe_divide(X, absmax, factor=1)

        if self._output_range not in (False, True):  # True = (-1, +1)
            output_range = self._output_range

            if not self._input_is_positive_only:
                X = (X + 1) / 2
            X = kbackend.clip(X, 0, 1)

            X = output_range[0] + (X * (output_range[1] - output_range[0]))
        else:
            X = kbackend.clip(X, -1, 1)

        return X


###############################################################################


class GreaterThanZero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.greater(inputs, kbackend.constant(0))


class LessEqualThanZero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.less_equal(inputs, kbackend.constant(0))


class Divide(klayers.Layer):
    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `Divide` layer should be called on exactly 2 inputs")
        a, b = inputs
        return a / b


class SafeDivide(klayers.Layer):
    def __init__(self, *args, factor: float = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if factor is None:
            factor = kbackend.epsilon()
        self._factor = factor

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                "A `SafeDivide` layer should be called on exactly 2 inputs"
            )
        a, b = inputs
        return ibackend.safe_divide(a, b, factor=self._factor)


###############################################################################
class Repeat(klayers.Layer):
    """Repeats the input n times. Similar to Keras' `RepeatVector`,
    except that it works on any Tensor.

    Input shape: 2D tensor of shape `(num_samples, features)`.
    Output shape: 3D tensor of shape `(num_samples, n, features)`.

    Args:
        n: Integer, repetition factor.
    """

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        if not isinstance(n, int):
            raise TypeError(f"Expected an integer value for `n`, got {type(n)}.")

    def call(self, inputs, *_args, **_kwargs):
        dims = inputs.shape.rank  # number of axes in Tensor
        assert dims >= 2
        inputs = tf.expand_dims(inputs, 1)

        # Construct array [1, n, 1, ..., 1] for tf.tile
        multiples = [1] * dims
        multiples.insert(1, self.n)
        return tf.tile(inputs, tf.constant(multiples))


class ReduceMean(klayers.Layer):
    """Reduce input augmented along `axis=1` by taking the mean."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.reduce_mean(inputs, axis=1, keepdims=False)


class Reshape(klayers.Layer):
    """Layer that reshapes tensor to the shape specified on init."""

    def __init__(self, shape: ShapeTuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = shape

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.reshape(inputs, self._shape)


class AugmentationToBatchAxis(klayers.Layer):
    """Move augmentation from axis=1 to batch axis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        input_shape = ibackend.shape(inputs)
        output_shape = [-1] + input_shape[2:]  # type: ignore
        return tf.reshape(inputs, output_shape)


class AugmentationFromBatchAxis(klayers.Layer):
    """Move augmentation from batch axis to axis=1.

    Args:
        n: Factor of augmentation.
    """

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        if not isinstance(n, int):
            raise TypeError(f"Expected an integer value for `n`, got {type(n)}.")

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        input_shape = ibackend.shape(inputs)
        output_shape = [-1, self.n] + input_shape[1:]
        return tf.reshape(inputs, output_shape)


class MultiplyWithLinspace(klayers.Layer):
    def __init__(self, start, end, *args, n=1, axis=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = start
        self._end = end
        self._n = n
        self._axis = axis

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        linspace = self._start + (self._end - self._start) * (
            kbackend.arange(self._n, dtype=kbackend.floatx()) / self._n
        )

        # Make broadcastable.
        shape = np.ones(len(kbackend.int_shape(inputs)))
        shape[self._axis] = self._n
        linspace = kbackend.reshape(linspace, shape)
        return inputs * linspace


class AddGaussianNoise(klayers.Layer):
    "Add Gaussian noise to Tensor. Also applies to test phase."

    def __init__(self, *args, mean: float = 0.0, stddev: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.stddev = stddev

    def call(self, inputs: Tensor, *_args, seed=None, **_kwargs) -> Tensor:
        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=self.mean,
            stddev=self.stddev,
            dtype=inputs.dtype,
            seed=seed,
        )
        return tf.add(inputs, noise)


class ExtractConv2DPatches(klayers.Layer):
    def __init__(self, kernel_shape, depth, strides, rates, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_shape = kernel_shape
        self._depth = depth
        self._strides = strides
        self._rates = rates
        self._padding = padding

    def call(self, inputs, *_args, **_kwargs):
        return ibackend.extract_conv2d_patches(
            inputs, self._kernel_shape, self._strides, self._rates, self._padding
        )


class RunningMeans(klayers.Layer):
    """Layer used to keep track of a running mean."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateful = True

    def build(self, input_shape: Sequence[ShapeTuple]) -> None:
        means_shape, counts_shape = input_shape

        self.means = self.add_weight(
            shape=means_shape, initializer="zeros", name="means", trainable=False
        )
        self.counts = self.add_weight(
            shape=counts_shape, initializer="zeros", name="counts", trainable=False
        )
        self.built = True

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> list[Tensor]:
        if len(inputs) != 2:
            raise ValueError(
                "A `RunningMeans` layer should be called on exactly 2 inputs"
            )
        means, counts = inputs
        new_counts = counts + self.counts

        # If new_means are not used for the model output,
        # the following part of the code will be executed after
        # self.counts is updated, therefore we cannot use it
        # hereafter.
        factor_new = ibackend.safe_divide(counts, new_counts, factor=1)
        factor_old = kbackend.ones_like(factor_new) - factor_new
        new_means = self.means * factor_old + means * factor_new

        # Update state.
        self.add_update(
            [
                kbackend.update(self.means, new_means),
                kbackend.update(self.counts, new_counts),
            ]
        )

        return [new_means, new_counts]


class Broadcast(klayers.Layer):
    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `Broadcast` layer should be called on exactly 2 inputs")
        target_shapped, x = inputs
        return target_shapped * 0 + x


class MaxNeuronSelection(klayers.Layer):
    """Applied to the last layer of a model, this reduces the output
    to the max neuron activation."""

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.reduce_max(
            inputs, axis=-1, keepdims=False
        )  # max along batch axis


class MaxNeuronIndex(klayers.Layer):
    """Applied to the last layer of a model, this reduces the output
    to the index of the max activated neuron."""

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.argmax(inputs, axis=-1)  # max along batch axis


class NeuronSelection(klayers.Layer):
    """Applied to the last layer of a model, this selects output neurons at given indices
    by wrapping `tf.gather`."""

    def call(self, inputs: list[Tensor], *_args, **_kwargs):
        if len(inputs) != 2:
            raise ValueError(
                "A `NeuronSelection` layer should be called on exactly 2 inputs"
            )
        X, index = inputs
        index_shape = ibackend.shape(index)
        if len(index_shape) != 2 or index_shape[1] != 2:
            raise ValueError(
                "Layer `NeuronSelection` expects index of shape (batch_size, 2),",
                f"got {index_shape}.",
            )
        return tf.gather_nd(X, index)
