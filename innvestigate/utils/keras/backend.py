# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import zip


###############################################################################
###############################################################################
###############################################################################


import keras.backend as K


__all__ = [
    "to_floatx",
    "gradients",
    "is_not_finite",
    "extract_conv2d_patches",
    "gather",
    "gather_nd",
]


###############################################################################
###############################################################################
###############################################################################


def to_floatx(x):
    return K.cast(x, K.floatx())


###############################################################################
###############################################################################
###############################################################################


def gradients(Xs, Ys, known_Ys):
    """Partial derivatives

    Computes the partial derivatives between Ys and Xs and
    using the gradients for Ys known_Ys.

    :param Xs: List of input tensors.
    :param Ys: List of output tensors that depend on Xs.
    :param known_Ys: Gradients for Ys.
    :return: Gradients for Xs given known_Ys
    """
    backend = K.backend()
    if backend == "theano":
        # no global import => do not break if module is not present
        assert len(Ys) == 1
        import theano.gradient
        known_Ys = {k: v for k, v in zip(Ys, known_Ys)}
        # todo: check the stop gradient issue here!
        return theano.gradient.grad(K.sum(Ys[0]), Xs, known_grads=known_Ys)
    elif backend == "tensorflow":
        # no global import => do not break if module is not present
        import tensorflow
        return tensorflow.gradients(Ys, Xs,
                                    grad_ys=known_Ys,
                                    stop_gradients=Xs)
    else:
        # todo: add cntk
        raise NotImplementedError()


###############################################################################
###############################################################################
###############################################################################


def is_not_finite(x):
    """Checks if tensor x is finite, if not throws an exception."""

    backend = K.backend()
    if backend == "theano":
        # no global import => do not break if module is not present
        import theano.tensor
        return theano.tensor.or_(theano.tensor.isnan(x),
                                 theano.tensor.isinf(x))
    elif backend == "tensorflow":
        # no global import => do not break if module is not present
        import tensorflow
        #x = tensorflow.check_numerics(x, "innvestigate - is_finite check")
        return tensorflow.logical_not(tensorflow.is_finite(x))
    else:
        # todo: add cntk
        raise NotImplementedError()


###############################################################################
###############################################################################
###############################################################################


def extract_conv2d_patches(x, kernel_shape, strides, rates, padding):
    """Extracts conv2d patches like TF function extract_image_patches.

    :param x: Input image.
    :param kernel_shape: Shape of the Keras conv2d kernel.
    :param strides: Strides of the Keras conv2d layer.
    :param rates: Dilation rates of the Keras conv2d layer.
    :param padding: Paddings of the Keras conv2d layer.
    :return: The extracted patches.
    """

    backend = K.backend()
    if backend == "theano":
        # todo: add theano function.
        raise NotImplementedError()
    elif backend == "tensorflow":
        # no global import => do not break if module is not present
        import tensorflow

        if K.image_data_format() == "channels_first":
            x = K.permute_dimensions(x, (0, 2, 3, 1))
        kernel_shape = [1, kernel_shape[0], kernel_shape[1], 1]
        strides = [1, strides[0], strides[1], 1]
        rates = [1, rates[0], rates[1], 1]
        ret = tensorflow.extract_image_patches(x,
                                               kernel_shape,
                                               strides,
                                               rates,
                                               padding.upper())

        if K.image_data_format() == "channels_first":
            # todo: check if we need to permute again.xs
            pass
        return ret
    else:
        # todo: add cntk
        raise NotImplementedError()


###############################################################################
###############################################################################
###############################################################################


def gather(x, axis, indices):
    """Works as TensorFlow's gather."""
    backend = K.backend()
    if backend == "theano":
        # todo: add theano function.
        raise NotImplementedError()
    elif backend == "tensorflow":
        # no global import => do not break if module is not present
        import tensorflow

        return tensorflow.gather(x, indices, axis=axis)
    else:
        # todo: add cntk
        raise NotImplementedError()


def gather_nd(x, indices):
    """Works as TensorFlow's gather_nd."""
    backend = K.backend()
    if backend == "theano":
        # todo: add theano function.
        raise NotImplementedError()
    elif backend == "tensorflow":
        # no global import => do not break if module is not present
        import tensorflow

        return tensorflow.gather_nd(x, indices)
    else:
        # todo: add cntk
        raise NotImplementedError()
