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


import keras.backend as K


__all__ = [
    "gradients",
    "is_not_finite",
]


###############################################################################
###############################################################################
###############################################################################


def gradients(Xs, Ys, known_Ys):
    "Partial derivates."
    backend = K.backend()
    if backend == "theano":
        # no global import => do not break if module is not present
        assert len(Ys) == 1
        import theano.gradient
        known_Ys = {k:v for k, v in zip(Ys, known_Ys)}
        return theano.gradient.grad(K.sum(Ys[0]), Xs, known_grads=known_Ys)
    elif backend == "tensorflow":
        # no global import => do not break if module is not present
        import tensorflow
        return tensorflow.gradients(Ys, Xs, grad_ys=known_Ys)
    else:
        # todo: add cntk
        raise NotImplementedError()
    pass


###############################################################################
###############################################################################
###############################################################################


def is_not_finite(x):
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
    pass
