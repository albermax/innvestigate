from __future__ import annotations

from builtins import range
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.keras.utils as kutils

import innvestigate.utils as iutils
import innvestigate.utils.keras.backend as ibackend
from innvestigate.utils.types import OptionalList, ShapeTuple, Tensor

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
    "Reshape",
    "MultiplyWithLinspace",
    "ExtractConv2DPatches",
    "RunningMeans",
    "Broadcast",
    "MaxNeuronSelection",
    "NeuronSelection",
]


###############################################################################


class OnesLike(klayers.Layer):
    """Create list of all-ones tensors of the same shapes as provided tensors."""

    def call(self, x: OptionalList[Tensor], **_kwargs) -> List[Tensor]:
        return [kbackend.ones_like(tmp) for tmp in iutils.to_list(x)]


class AsFloatX(klayers.Layer):
    def call(self, x: OptionalList[Tensor], **_kwargs) -> List[Tensor]:
        return [ibackend.cast_to_floatx(tmp) for tmp in iutils.to_list(x)]


class FiniteCheck(klayers.Layer):
    def call(self, Xs: OptionalList[Tensor], **_kwargs) -> List[Tensor]:
        return [
            kbackend.sum(ibackend.cast_to_floatx(ibackend.is_not_finite(X)))
            for X in iutils.to_list(Xs)
        ]


###############################################################################


class _Reduce(klayers.Layer):
    def __init__(
        self,
        axis: Optional[OptionalList[int]] = -1,
        keepdims: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(*args, **kwargs)

    def call(self, x: OptionalList[Tensor]) -> Tensor:
        return self._apply_reduce(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape: ShapeTuple) -> ShapeTuple:
        if self.axis is None:
            if self.keepdims is False:
                return (1,)
            return tuple(np.ones_like(input_shape))  # type: ignore
        else:
            axes = np.arange(len(input_shape))
            if self.keepdims is False:
                for i in iutils.to_list(self.axis):
                    axes = np.delete(axes, i, 0)
            else:
                for i in iutils.to_list(self.axis):
                    axes[i] = 1
            return tuple([idx for i, idx in enumerate(input_shape) if i in axes])

    def _apply_reduce(
        self, x: Tensor, axis: Optional[OptionalList[int]], keepdims: bool
    ) -> Tensor:
        raise NotImplementedError()


class Sum(_Reduce):
    def _apply_reduce(
        self, x: Tensor, axis: Optional[OptionalList[int]], keepdims: bool
    ) -> Tensor:
        return kbackend.sum(x, axis=axis, keepdims=keepdims)


###############################################################################


class _Map(klayers.Layer):
    def call(self, X: OptionalList[Tensor]) -> OptionalList[Tensor]:
        if isinstance(X, list) and len(X) == 1:
            X = X[0]
        return self._apply_map(X)

    def compute_output_shape(self, input_shape: ShapeTuple) -> ShapeTuple:
        return input_shape

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
        self, min_value: Union[float, int, Tensor], max_value: Union[float, int, Tensor]
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
        dims: Tuple[int] = kbackend.int_shape(X)
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
    def call(self, x: Tensor) -> Tensor:
        return kbackend.greater(x, kbackend.constant(0))


class LessEqualThanZero(klayers.Layer):
    def call(self, x: Tensor) -> Tensor:
        return kbackend.less_equal(x, kbackend.constant(0))


class Divide(klayers.Layer):
    def call(self, inputs: List[Tensor]) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `Divide` layer should be called on exactly 2 inputs")
        a, b = inputs
        return a / b

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return input_shapes[0]


class SafeDivide(klayers.Layer):
    def __init__(self, *args, factor: float = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if factor is None:
            factor = kbackend.epsilon()
        self._factor = factor

    def call(self, inputs: List[Tensor]) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                "A `SafeDivide` layer should be called on exactly 2 inputs"
            )
        a, b = inputs
        return ibackend.safe_divide(a, b, factor=self._factor)

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return input_shapes[0]


###############################################################################


class Reshape(klayers.Layer):
    """Layer that reshapes tensor to the shape specified on init."""

    def __init__(self, shape: ShapeTuple, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._shape = shape

    def call(self, X: Tensor) -> Tensor:
        return kbackend.reshape(X, self._shape)

    def compute_output_shape(self, _input_shapes) -> ShapeTuple:
        return tuple(
            dim if (dim is not None and dim >= 0) else None for dim in self._shape
        )


class MultiplyWithLinspace(klayers.Layer):
    def __init__(self, start, end, n=1, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = start
        self._end = end
        self._n = n
        self._axis = axis

    def call(self, x: Tensor) -> Tensor:
        linspace = self._start + (self._end - self._start) * (
            kbackend.arange(self._n, dtype=kbackend.floatx()) / self._n
        )

        # Make broadcastable.
        shape = np.ones(len(kbackend.int_shape(x)))
        shape[self._axis] = self._n
        linspace = kbackend.reshape(linspace, shape)
        return x * linspace

    def compute_output_shape(self, input_shapes: ShapeTuple) -> ShapeTuple:
        return (
            input_shapes[: self._axis]
            + (max(self._n, input_shapes[self._axis]),)
            + input_shapes[self._axis + 1 :]
        )


class ExtractConv2DPatches(klayers.Layer):
    def __init__(self, kernel_shape, depth, strides, rates, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_shape = kernel_shape
        self._depth = depth
        self._strides = strides
        self._rates = rates
        self._padding = padding

    def call(self, x):
        return ibackend.extract_conv2d_patches(
            x, self._kernel_shape, self._strides, self._rates, self._padding
        )

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        if kbackend.image_data_format() == "channels_first":
            space = input_shapes[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = kutils.conv_utils.conv_output_length(
                    space[i],
                    self._kernel_shape[i],
                    padding=self._padding,
                    stride=self._strides[i],
                    dilation=self._rates[i],
                )
                new_space.append(new_dim)

        if kbackend.image_data_format() == "channels_last":
            space = input_shapes[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = kutils.conv_utils.conv_output_length(
                    space[i],
                    self._kernel_shape[i],
                    padding=self._padding,
                    stride=self._strides[i],
                    dilation=self._rates[i],
                )
                new_space.append(new_dim)

        return (
            (input_shapes[0],)
            + tuple(new_space)
            + (np.product(self._kernel_shape) * self._depth,)
        )


class RunningMeans(klayers.Layer):
    """Layer used to keep track of a running mean."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateful = True

    def build(self, input_shapes: Sequence[ShapeTuple]) -> None:
        means_shape, counts_shape = input_shapes

        self.means = self.add_weight(
            shape=means_shape, initializer="zeros", name="means", trainable=False
        )
        self.counts = self.add_weight(
            shape=counts_shape, initializer="zeros", name="counts", trainable=False
        )
        self.built = True

    def call(self, inputs: List[Tensor]) -> List[Tensor]:
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

    def compute_output_shape(self, input_shapes: ShapeTuple) -> ShapeTuple:
        return input_shapes


class Broadcast(klayers.Layer):
    def call(self, inputs: List[Tensor]) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `Broadcast` layer should be called on exactly 2 inputs")
        target_shapped, x = inputs
        return target_shapped * 0 + x

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return input_shapes[0]


class MaxNeuronSelection(klayers.Layer):
    """Applied to the last layer of a model, this reduces the output
    to the max neuron activation."""

    def call(self, x: Tensor) -> Tensor:
        return kbackend.max(x)


class NeuronSelection(klayers.Layer):
    """Applied to the last layer of a model, this selects output neurons at given indices
    by wrapping `tf.gather`."""

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError(
                "A `NeuronSelection` layer should be called on exactly 2 inputs"
            )
        x, indices = inputs
        return tf.gather(x, indices, axis=1)
