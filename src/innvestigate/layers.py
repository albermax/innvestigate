from __future__ import annotations

from builtins import range, zip
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.python.keras.utils as kutils

import innvestigate.utils as iutils
import innvestigate.utils.keras.backend as ibackend
from innvestigate.utils.types import OptionalList, ShapeTuple, Tensor

__all__ = [
    "OnesLike",
    "AsFloatX",
    "FiniteCheck",
    "Gradient",
    "GradientWRT",
    "Min",
    "Max",
    "GreaterThanZero",
    "LessEqualThanZero",
    "Sum",
    "Mean",
    "CountNonZero",
    "Identity",
    "Abs",
    "Square",
    "Clip",
    "Project",
    "Transpose",
    "Dot",
    "SafeDivide",
    "Repeat",
    "Reshape",
    "MultiplyWithLinspace",
    "TestPhaseGaussianNoise",
    "ExtractConv2DPatches",
    "RunningMeans",
    "Broadcast",
    "GatherND",
]


###############################################################################


class OnesLike(klayers.Layer):
    """Create list of all-ones tensors of the same shapes as provided tensors."""

    def call(self, x: OptionalList[Tensor], **_kwargs) -> List[Tensor]:
        return [kbackend.ones_like(tmp) for tmp in iutils.to_list(x)]


class AsFloatX(klayers.Layer):
    def call(self, x: OptionalList[Tensor], **_kwargs) -> List[Tensor]:
        return [kbackend.cast_to_floatx(tmp) for tmp in iutils.to_list(x)]


class FiniteCheck(klayers.Layer):
    def call(self, x: OptionalList[Tensor], **_kwargs) -> List[Tensor]:
        return [
            kbackend.sum(kbackend.cast_to_floatx(kbackend.is_not_finite(tmp)))
            for tmp in iutils.to_list(x)
        ]


###############################################################################


class Gradient(klayers.Layer):
    "Returns gradient of sum(output), expects inputs+[output,]."

    def call(self, x: List[Tensor]) -> List[Tensor]:
        inputs, output = x[:-1], x[-1]
        return kbackend.gradients(kbackend.sum(output), inputs)  # type: ignore

    def compute_output_shape(self, input_shapes: List[ShapeTuple]) -> List[ShapeTuple]:
        return input_shapes[:-1]


class GradientWRT(klayers.Layer):
    """Returns gradient wrt to another layer and given gradient,
    expects inputs+[output,]."""

    # TODO: add documentation

    def __init__(
        self, n_inputs: int, mask: Optional[List[bool]] = None, **kwargs
    ) -> None:
        self.n_inputs = n_inputs
        self.mask = mask
        super().__init__(**kwargs)

    def call(self, x: List[Tensor]) -> List[Tensor]:
        assert isinstance(x, list)
        Xs, tmp_Ys = x[: self.n_inputs], x[self.n_inputs :]
        assert len(tmp_Ys) % 2 == 0
        len_Ys = len(tmp_Ys) // 2
        Ys, known_Ys = tmp_Ys[:len_Ys], tmp_Ys[len_Ys:]
        ret = iK.gradients(Xs, Ys, known_Ys)
        if self.mask is not None:
            ret = [x for c, x in zip(self.mask, ret) if c]
        self.__workaround__len_ret = len(ret)
        return ret

    def compute_output_shape(self, input_shapes: List[ShapeTuple]) -> List[ShapeTuple]:
        if self.mask is None:
            return input_shapes[: self.n_inputs]
        else:
            return [
                shape
                for shape, keep in zip(input_shapes[: self.n_inputs], self.mask)
                if keep
            ]

    # TODO: remove once keras is fixed.
    # this is a workaround for cases when
    # wrapper and skip connections are used together.
    # bring the fix into keras and remove once
    # keras is patched.
    def compute_mask(self, inputs, mask=None):
        """Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """
        if not self.supports_masking:
            if not isinstance(mask, list) or any(m is not None for m in mask):
                raise TypeError(
                    f"Layer {self.name} does not support masking, ",
                    f"but was passed an input_mask: {str(mask)}",
                )
            # masking not explicitly supported: return None as mask

            # this is the workaround for model.run_internal_graph.
            # it is required that there as many masks as outputs:
            return [None for _ in range(self.__workaround__len_ret)]
        # if masking is explicitly supported, by default
        # carry over the input mask
        return mask


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
        super(_Reduce, self).__init__(*args, **kwargs)

    def call(self, x: OptionalList[Tensor]) -> Tensor:
        return self._apply_reduce(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape: ShapeTuple) -> ShapeTuple:
        if self.axis is None:
            if self.keepdims is False:
                return (1,)
            else:
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


class Min(_Reduce):
    def _apply_reduce(
        self, x: Tensor, axis: Optional[OptionalList[int]], keepdims: bool
    ) -> Tensor:
        return kbackend.min(x, axis=axis, keepdims=keepdims)


class Max(_Reduce):
    def _apply_reduce(
        self, x: Tensor, axis: Optional[OptionalList[int]], keepdims: bool
    ) -> Tensor:
        return kbackend.max(x, axis=axis, keepdims=keepdims)


class Sum(_Reduce):
    def _apply_reduce(
        self, x: Tensor, axis: Optional[OptionalList[int]], keepdims: bool
    ) -> Tensor:
        return kbackend.sum(x, axis=axis, keepdims=keepdims)


class Mean(_Reduce):
    def _apply_reduce(
        self, x: Tensor, axis: Optional[OptionalList[int]], keepdims: bool
    ) -> Tensor:
        return kbackend.mean(x, axis=axis, keepdims=keepdims)


class CountNonZero(_Reduce):
    def _apply_reduce(
        self, x: Tensor, axis: Optional[OptionalList[int]], keepdims: bool
    ) -> Tensor:
        return kbackend.sum(
            kbackend.cast_to_floatx(kbackend.not_equal(x, kbackend.constant(0))),
            axis=axis,
            keepdims=keepdims,
        )


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
        return kbackend.identity(X)


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
        def safe_divide(A: Tensor, B: Tensor) -> Tensor:
            return A / (
                B + kbackend.cast_to_floatx(kbackend.equal(B, kbackend.constant(0))) * 1
            )

        dims: Tuple[int] = kbackend.int_shape(X)
        n_dim: int = len(dims)
        axes = tuple(range(1, n_dim))

        if len(axes) == 1:
            # TODO(albermax): this is only the case when the dimension in this
            # axis is 1, fix this.
            # Cannot reduce
            return X

        absmax = kbackend.max(kbackend.abs(X), axis=axes, keepdims=True)
        X = safe_divide(X, absmax)

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


class Transpose(klayers.Layer):
    def __init__(self, axes=None, **kwargs) -> None:
        self._axes = axes
        super().__init__(**kwargs)

    def call(self, x: Tensor) -> Tensor:
        if self._axes is None:
            return kbackend.transpose(x)
        else:
            return kbackend.permute_dimensions(x, self._axes)

    def compute_output_shape(self, input_shape: ShapeTuple) -> ShapeTuple:
        if self._axes is None:
            return input_shape[::-1]  # invert input shape
        else:
            return tuple(np.asarray(input_shape)[list(self._axes)])


class Dot(klayers.Layer):
    def call(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        a, b = x
        return kbackend.dot(a, b)

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return (input_shapes[0][0], input_shapes[1][1])


class Divide(klayers.Layer):
    def call(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        a, b = x
        return a / b

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return input_shapes[0]


class SafeDivide(klayers.Layer):
    def __init__(self, *args, factor: float = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if factor is None:
            factor = kbackend.epsilon()
        self._factor = factor

    def call(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        a, b = x
        return a / (
            b
            + kbackend.cast_to_floatx(kbackend.equal(b, kbackend.constant(0)))
            * self._factor
        )

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return input_shapes[0]


###############################################################################


class Repeat(klayers.Layer):
    def __init__(self, n: int, axis, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = n
        self._axis = axis

    def call(self, x: Tensor) -> Tensor:
        return kbackend.repeat_elements(x, self._n, self._axis)

    def compute_output_shape(
        self, input_shapes: OptionalList[ShapeTuple]
    ) -> ShapeTuple:
        input_shape: ShapeTuple

        if isinstance(input_shapes, list):
            input_shape = input_shapes[0]
        elif isinstance(input_shapes, tuple):
            input_shape = input_shapes
        else:
            raise TypeError(
                "Expected shape tuple (tuple of integers) or list of shape tuples."
            )

        if input_shape[0] is None:
            return input_shape
        else:
            return (input_shape[0] * self._n,) + input_shape[1:]


class Reshape(klayers.Layer):
    def __init__(self, shape: Iterable[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = shape

    def call(self, x: Tensor) -> Tensor:
        return kbackend.reshape(x, self._shape)

    def compute_output_shape(self, _input_shapes) -> ShapeTuple:
        return tuple(x if x >= 0 else None for x in self._shape)


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


class TestPhaseGaussianNoise(klayers.GaussianNoise):
    def call(self, inputs: Tensor) -> Tensor:
        # Always add Gaussian noise!
        return super().call(inputs, training=True)


class ExtractConv2DPatches(klayers.Layer):
    def __init__(self, kernel_shape, depth, strides, rates, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_shape = kernel_shape
        self._depth = depth
        self._strides = strides
        self._rates = rates
        self._padding = padding

    def call(self, x):
        return kbackend.extract_conv2d_patches(
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
    def __init__(self, *args, **kwargs) -> None:
        self.stateful = True
        super().__init__(*args, **kwargs)

    def build(self, input_shapes: Sequence[ShapeTuple]) -> None:
        means_shape, counts_shape = input_shapes

        self.means = self.add_weight(
            shape=means_shape, initializer="zeros", name="means", trainable=False
        )
        self.counts = self.add_weight(
            shape=counts_shape, initializer="zeros", name="counts", trainable=False
        )
        self.built = True

    def call(self, x: List[Tensor]) -> List[Tensor]:
        def safe_divide(a, b):
            return a / (
                b + kbackend.cast_to_floatx(kbackend.equal(b, kbackend.constant(0))) * 1
            )

        means, counts = x

        new_counts = counts + self.counts

        # If new_means are not used for the model output,
        # the following part of the code will be executed after
        # self.counts is updated, therefore we cannot use it
        # hereafter.
        factor_new = safe_divide(counts, new_counts)
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
    def call(self, x: List[Tensor]) -> Tensor:
        target_shapped, x = x
        return target_shapped * 0 + x

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return input_shapes[0]


class GatherND(klayers.Layer):
    def call(self, inputs):
        x, indices = inputs
        return tf.gather_nd(x, indices)

    def compute_output_shape(self, input_shapes: Sequence[ShapeTuple]) -> ShapeTuple:
        return input_shapes[1][:2] + input_shapes[0][2:]
