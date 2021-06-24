# Get Python six functionality:
from __future__ import absolute_import, division, print_function, unicode_literals

import keras.backend as K

from tests.networks import base

__all__ = [
    "log_reg",
    "mlp_2dense",
    "mlp_3dense",
    "cnn_1convb_2dense",
    "cnn_2convb_2dense",
    "cnn_2convb_3dense",
    "cnn_3convb_3dense",
]


###############################################################################
###############################################################################
###############################################################################


if K.image_data_format() == "channels_first":
    __input_shape__ = [None, 1, 28, 28]
else:
    __input_shape__ = [None, 28, 28, 1]
__output_n__ = 10


###############################################################################


def log_reg(activation=None):
    return base.log_reg(__input_shape__, __output_n__, activation=activation)


###############################################################################


def mlp_2dense(activation=None):
    return base.mlp_2dense(
        __input_shape__,
        __output_n__,
        activation=activation,
        dense_units=512,
        dropout_rate=0.25,
    )


def mlp_3dense(activation=None):
    return base.mlp_3dense(
        __input_shape__,
        __output_n__,
        activation=activation,
        dense_units=512,
        dropout_rate=0.25,
    )


###############################################################################


def cnn_1convb_2dense(activation=None):
    return base.cnn_1convb_2dense(
        __input_shape__,
        __output_n__,
        activation=activation,
        dense_units=512,
        dropout_rate=0.25,
    )


def cnn_2convb_2dense(activation=None):
    return base.cnn_2convb_2dense(
        __input_shape__,
        __output_n__,
        activation=activation,
        dense_units=512,
        dropout_rate=0.25,
    )


def cnn_2convb_3dense(activation=None):
    return base.cnn_2convb_3dense(
        __input_shape__,
        __output_n__,
        activation=activation,
        dense_units=512,
        dropout_rate=0.25,
    )


def cnn_3convb_3dense(activation=None):
    return base.cnn_3convb_3dense(
        __input_shape__,
        __output_n__,
        activation=activation,
        dense_units=512,
        dropout_rate=0.25,
    )
