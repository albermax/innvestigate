# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import range, zip


###############################################################################
###############################################################################
###############################################################################

import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as keras_layers
from tensorflow.python.keras.utils import conv_utils
import numpy as np


from . import utils as iutils
from .utils.keras import backend as iK


__all__ = [
    "Constant",
    "Zero",
    "One",
    "ZerosLike",
    "OnesLike",
    "AsFloatX",
    "FiniteCheck",

    "Gradient",
    "GradientWRT",

    "Min",
    "Max",
    "Greater",
    "Less",
    "GreaterThanZero",
    "LessThanZero",
    "GreaterEqual",
    "LessEqual",
    "GreaterEqualThanZero",
    "LessEqualThanZero",
    "Sum",
    "Mean",
    "CountNonZero",

    "Identity",
    "Abs",
    "Square",
    "Clip",
    "Project",
    "Print",

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
    "Gather",
    "GatherND",
]


###############################################################################
###############################################################################
###############################################################################


def Constant(c, reference=None):
    if reference is None:
        return K.constant(c)
    else:
        dtype = K.dtype(reference)
        return K.constant(np.dtype(dtype)(c), dtype=dtype)


def Zero(reference=None):
    return Constant(0, reference=reference)


def One(reference=None):
    return Constant(1, reference=reference)


class ZerosLike(keras_layers.Layer):
    def call(self, x):
        return [K.zeros_like(tmp) for tmp in iutils.to_list(x)]


class OnesLike(keras_layers.Layer):
    def call(self, x):
        return K.ones_like(x)#[K.ones_like(tmp) for tmp in iutils.to_list(x)]


class AsFloatX(keras_layers.Layer):
    def call(self, x):
        return [iK.to_floatx(tmp) for tmp in iutils.to_list(x)]


class FiniteCheck(keras_layers.Layer):
    def call(self, x):
        return [K.sum(iK.to_floatx(iK.is_not_finite(tmp)))
                for tmp in iutils.to_list(x)]


###############################################################################
###############################################################################
###############################################################################


class Gradient(keras_layers.Layer):
    "Returns gradient of sum(output), expects inputs+[output,]."

    def call(self, x):
        inputs, output = x[:-1], x[-1]
        return K.gradients(K.sum(output), inputs)

    def compute_output_shape(self, input_shapes):
        return input_shapes[:-1]


class GradientWRT(keras_layers.Layer):
    "Returns gradient wrt to another layer and given gradient,"
    " expects inputs+[output,]."

    def __init__(self, n_inputs, mask=None, **kwargs):
        self.n_inputs = n_inputs
        self.mask = mask
        super(GradientWRT, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        Xs, tmp_Ys = x[:self.n_inputs], x[self.n_inputs:]
        assert len(tmp_Ys) % 2 == 0
        len_Ys = len(tmp_Ys) // 2
        Ys, known_Ys = tmp_Ys[:len_Ys], tmp_Ys[len_Ys:]

        if len(Ys) == 1:
            Ys = Ys[0]
        if len(known_Ys) == 1:
            known_Ys = known_Ys[0]

        ret = tensorflow.gradients(Ys, Xs,
                                   grad_ys=known_Ys,
                                   stop_gradients=Xs)

        if self.mask is not None:
            ret = [x for c, x in zip(self.mask, ret) if c]
        self.__workaround__len_ret = len(ret)
        return ret

    def compute_output_shape(self, input_shapes):
        if self.mask is None:
            return input_shapes[:self.n_inputs]
        else:
            return [x for c, x in zip(self.mask, input_shapes[:self.n_inputs])
                    if c]

    # todo: remove once keras is fixed.
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
            if mask is not None:
                if isinstance(mask, list):
                    if any(m is not None for m in mask):
                        raise TypeError('Layer ' + self.name +
                                        ' does not support masking, '
                                        'but was passed an input_mask: ' +
                                        str(mask))
                else:
                    raise TypeError('Layer ' + self.name +
                                    ' does not support masking, '
                                    'but was passed an input_mask: ' +
                                    str(mask))
            # masking not explicitly supported: return None as mask

            # this is the workaround for model.run_internal_graph.
            # it is required that there as many masks as outputs:
            return [None for _ in range(self.__workaround__len_ret)]
        # if masking is explicitly supported, by default
        # carry over the input mask
        return mask


###############################################################################
###############################################################################
###############################################################################


class _Reduce(keras_layers.Layer):

    def __init__(self, axis=-1, keepdims=False, *args, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super(_Reduce, self).__init__(*args, **kwargs)

    def call(self, x):
        return self._apply_reduce(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):
        if self.axis is None:
            if self.keepdims is False:
                return (1,)
            else:
                return tuple(np.ones_like(input_shape))
        else:
            axes = np.arange(len(input_shape))
            if self.keepdims is False:
                for i in iutils.to_list(self.axis):
                    axes = np.delete(axes, i, 0)

            else:
                for i in iutils.to_list(self.axis):
                    axes[i] = 1
            return tuple([idx
                          for i, idx in enumerate(input_shape)
                          if i in axes])

    def _apply_reduce(self, x, axis, keepdims):
        raise NotImplementedError()


class Min(_Reduce):
    def _apply_reduce(self, x, axis, keepdims):
        return K.min(x, axis=axis, keepdims=keepdims)


class Max(_Reduce):
    def _apply_reduce(self, x, axis, keepdims):
        return K.max(x, axis=axis, keepdims=keepdims)


class Sum(_Reduce):
    def _apply_reduce(self, x, axis, keepdims):
        return K.sum(x, axis=axis, keepdims=keepdims)


class Mean(_Reduce):
    def _apply_reduce(self, x, axis, keepdims):
        return K.mean(x, axis=axis, keepdims=keepdims)


class CountNonZero(_Reduce):
    def _apply_reduce(self, x, axis, keepdims):
        return K.sum(iK.to_floatx(K.not_equal(x, K.constant(0))),
                     axis=axis,
                     keepdims=keepdims)


###############################################################################
###############################################################################
###############################################################################


class _Map(keras_layers.Layer):

    def call(self, x):
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        return self._apply_map(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _apply_map(self, x):
        raise NotImplementedError()


class Identity(_Map):
    def _apply_map(self, x):
        return x


class Abs(_Map):
    def _apply_map(self, x):
        return K.abs(x)


class Square(_Map):
    def _apply_map(self, x):
        return K.square(x)


class Clip(_Map):

    def __init__(self, min_value, max_value):
        self._min_value = min_value
        self._max_value = max_value
        return super(Clip, self).__init__()

    def _apply_map(self, x):
        return K.clip(x, self._min_value, self._max_value)


class Project(_Map):

    def __init__(self, output_range=False, input_is_postive_only=False):
        self._output_range = output_range
        self._input_is_positive_only = input_is_postive_only
        return super(Project, self).__init__()

    def _apply_map(self, x):
        def safe_divide(a, b):
            return a / (b + iK.to_floatx(K.equal(b, K.constant(0))) * 1)

        dims = K.int_shape(x)
        n_dim = len(dims)
        axes = tuple(range(1, n_dim))
        if len(axes) == 1:
            # TODO(albermax): this is only the case when the dimension in this
            # axis is 1, fix this.
            # Cannot reduce
            return x

        absmax = K.max(K.abs(x),
                       axis=axes,
                       keepdims=True)
        x = safe_divide(x, absmax)

        if self._output_range not in (False, True):  # True = (-1, +1)
            output_range = self._output_range

            if not self._input_is_positive_only:
                x = (x+1) / 2
            x = K.clip(x, 0, 1)

            x = output_range[0] + (x * (output_range[1]-output_range[0]))
        else:
            x = K.clip(x, -1, 1)

        return x


class Print(_Map):
    def _apply_map(self, x):
        return K.print_tensor(x)


###############################################################################
###############################################################################
###############################################################################


class Greater(keras_layers.Layer):
    def call(self, x):
        a, b = x
        return K.greater(a, b)


class Less(keras_layers.Layer):
    def call(self, x):
        a, b = x
        return K.less(a, b)


class GreaterThanZero(keras_layers.Layer):
    def call(self, x):
        return K.greater(x, K.constant(0))


class LessThanZero(keras_layers.Layer):
    def call(self, x):
        return K.less(x, K.constant(0))


class GreaterEqual(keras_layers.Layer):
    def call(self, x):
        a, b = x
        return K.greater_equal(a, b)


class LessEqual(keras_layers.Layer):
    def call(self, x):
        a, b = x
        return K.less_equal(a, b)


class GreaterEqualThanZero(keras_layers.Layer):
    def call(self, x):
        return K.greater_equal(x, K.constant(0))


class LessEqualThanZero(keras_layers.Layer):
    def call(self, x):
        return K.less_equal(x, K.constant(0))


class Transpose(keras_layers.Layer):

    def __init__(self, axes=None, **kwargs):
        self._axes = axes
        super(Transpose, self).__init__(**kwargs)

    def call(self, x):
        if self._axes is None:
            return K.transpose(x)
        else:
            return K.permute_dimensions(x, self._axes)

    def compute_output_shape(self, input_shape):
        if self._axes is None:
            return input_shape[::-1]
        else:
            return tuple(np.asarray(input_shape)[list(self._axes)])


class Dot(keras_layers.Layer):

    def call(self, x):
        a, b = x
        return K.dot(a, b)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[1][1])


class Divide(keras_layers.Layer):

    def call(self, x):
        a, b = x
        return a / b

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


class SafeDivide(keras_layers.Layer):

    def __init__(self, *args, **kwargs):
        factor = kwargs.pop("factor", None)
        if factor is None:
            factor = K.epsilon()
        self._factor = factor

        return super(SafeDivide, self).__init__(*args, **kwargs)

    def call(self, x):
        a, b = x
        return a / (b + iK.to_floatx(K.equal(b, K.constant(0))) * self._factor)

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


###############################################################################
###############################################################################
###############################################################################


class Repeat(keras_layers.Layer):

    def __init__(self, n, axis, *args, **kwargs):
        self._n = n
        self._axis = axis
        return super(Repeat, self).__init__(*args, **kwargs)

    def call(self, x):
        return K.repeat_elements(x, self._n, self._axis)

    def compute_output_shape(self, input_shapes):
        if isinstance(input_shapes, list):
            input_shape = input_shapes[0]
        else:
            input_shape = input_shapes

        if input_shape[0] is None:
            return input_shape
        else:
            return (input_shape[0]*self._n,)+input_shape[1:]


class Reshape(keras_layers.Layer):

    def __init__(self, shape, *args, **kwargs):
        self._shape = shape
        return super(Reshape, self).__init__(*args, **kwargs)

    def call(self, x):
        return K.reshape(x, self._shape)

    def compute_output_shape(self, input_shapes):
        return tuple(x if x >= 0 else None for x in self._shape)


class MultiplyWithLinspace(keras_layers.Layer):

    def __init__(self, start, end, n=1, axis=-1, *args, **kwargs):
        self._start = start
        self._end = end
        self._n = n
        self._axis = axis
        return super(MultiplyWithLinspace, self).__init__(*args, **kwargs)

    def call(self, x):
        linspace = (self._start +
                    (self._end-self._start) *
                    (K.arange(self._n, dtype=K.floatx())/self._n))

        # Make broadcastable.
        shape = np.ones(len(K.int_shape(x)))
        shape[self._axis] = self._n
        linspace = K.reshape(linspace, shape)
        return x * linspace

    def compute_output_shape(self, input_shapes):
        ret = input_shapes[:]
        ret = (ret[:self._axis] +
               (max(self._n, ret[self._axis]),) +
               ret[self._axis+1:])
        return ret


class TestPhaseGaussianNoise(keras_layers.GaussianNoise):

    def call(self, inputs):
        # Always add Gaussian noise!
        return super(TestPhaseGaussianNoise, self).call(inputs, training=True)


class ExtractConv2DPatches(keras_layers.Layer):

    def __init__(self,
                 kernel_shape,
                 depth,
                 strides,
                 rates,
                 padding,
                 *args,
                 **kwargs):
        self._kernel_shape = kernel_shape
        self._depth = depth
        self._strides = strides
        self._rates = rates
        self._padding = padding
        return super(ExtractConv2DPatches, self).__init__(*args, **kwargs)

    def call(self, x):
        return iK.extract_conv2d_patches(x,
                                         self._kernel_shape,
                                         self._strides,
                                         self._rates,
                                         self._padding)

    def compute_output_shape(self, input_shapes):
        if K.image_data_format() == 'channels_first':
            space = input_shapes[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self._kernel_shape[i],
                    padding=self._padding,
                    stride=self._strides[i],
                    dilation=self._rates[i])
                new_space.append(new_dim)

        if K.image_data_format() == 'channels_last':
            space = input_shapes[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self._kernel_shape[i],
                    padding=self._padding,
                    stride=self._strides[i],
                    dilation=self._rates[i])
                new_space.append(new_dim)

        return ((input_shapes[0],) +
                tuple(new_space) +
                (np.product(self._kernel_shape) * self._depth,))


class RunningMeans(keras_layers.Layer):

    def __init__(self, *args, **kwargs):
        self.stateful = True
        super(RunningMeans, self).__init__(*args, **kwargs)

    def build(self, input_shapes):
        means_shape, counts_shape = input_shapes

        self.means = self.add_weight(shape=means_shape,
                                     initializer="zeros",
                                     name="means",
                                     trainable=False)
        self.counts = self.add_weight(shape=counts_shape,
                                      initializer="zeros",
                                      name="counts",
                                      trainable=False)
        self.built = True

    def call(self, x):
        def safe_divide(a, b):
            return a / (b + iK.to_floatx(K.equal(b, K.constant(0))) * 1)

        means, counts = x

        new_counts = counts + self.counts

        # If new_means are not used for the model output,
        # the following part of the code will be executed after
        # self.counts is updated, therefore we cannot use it
        # hereafter.
        factor_new = safe_divide(counts, new_counts)
        factor_old = K.ones_like(factor_new) - factor_new
        new_means = self.means * factor_old + means * factor_new

        # Update state.
        self.add_update([
            K.update(self.means, new_means),
            K.update(self.counts, new_counts),
        ])

        return [new_means, new_counts]

    def compute_output_shape(self, input_shapes):
        return input_shapes


class Broadcast(keras_layers.Layer):

    def call(self, x):
        target_shapped, x = x
        return target_shapped * 0 + x

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


class Gather(keras_layers.Layer):

    def call(self, inputs):
        x, indices = inputs
        return tensorflow.gather(x, indices, axis=1)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[1][0])+input_shapes[0][2:]


class GatherND(keras_layers.Layer):

    def call(self, inputs):
        x, indices = inputs
        a = tensorflow.gather_nd(x, indices)
        # Workaround to not break code when moving to tf 2.1
        b = tensorflow.zeros_like(a)
        return a + b

    def compute_output_shape(self, input_shapes):
        return input_shapes[1][:2]+input_shapes[0][2:]
