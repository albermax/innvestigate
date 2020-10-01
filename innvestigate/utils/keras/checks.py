# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import inspect
# import tensorflow.keras.engine.topology
import tensorflow.keras.layers


# Prevents circular imports.
def get_kgraph():
    from . import graph as kgraph
    return kgraph


__all__ = [
    "get_current_layers",
    "get_known_layers",
    "get_activation_search_safe_layers",

    "contains_activation",
    "contains_kernel",
    "only_relu_activation",
    "is_network",
    "is_convnet_layer",
    "is_relu_convnet_layer",
    "is_average_pooling",
    "is_max_pooling",
    "is_input_layer",
    "is_batch_normalization_layer",
]


###############################################################################
###############################################################################
###############################################################################


def get_current_layers():
    """
    Returns a list of currently available layers in Keras.
    """
    class_set = set([(getattr(tensorflow.keras.layers, name), name)
                     for name in dir(tensorflow.keras.layers)
                     if (inspect.isclass(getattr(tensorflow.keras.layers, name)) and
                         issubclass(getattr(tensorflow.keras.layers, name),
                                    tensorflow.keras.layers.Layer))])
    return [x[1] for x in sorted((str(x[0]), x[1]) for x in class_set)]


def get_known_layers():
    """
    Returns a list of tensorflow.keras layer we are aware of.
    """

    # Inside function to not break import if Keras changes.
    KNOWN_LAYERS = (
        tensorflow.keras.layers.InputLayer,
        tensorflow.keras.layers.ELU,
        tensorflow.keras.layers.LeakyReLU,
        tensorflow.keras.layers.PReLU,
        tensorflow.keras.layers.Softmax,
        tensorflow.keras.layers.ThresholdedReLU,
        tensorflow.keras.layers.Conv1D,
        tensorflow.keras.layers.Conv2D,
        tensorflow.keras.layers.Conv2DTranspose,
        tensorflow.keras.layers.Conv3D,
        tensorflow.keras.layers.Conv3DTranspose,
        tensorflow.keras.layers.Cropping1D,
        tensorflow.keras.layers.Cropping2D,
        tensorflow.keras.layers.Cropping3D,
        tensorflow.keras.layers.SeparableConv1D,
        tensorflow.keras.layers.SeparableConv2D,
        tensorflow.keras.layers.UpSampling1D,
        tensorflow.keras.layers.UpSampling2D,
        tensorflow.keras.layers.UpSampling3D,
        tensorflow.keras.layers.ZeroPadding1D,
        tensorflow.keras.layers.ZeroPadding2D,
        tensorflow.keras.layers.ZeroPadding3D,
        tensorflow.keras.layers.ConvLSTM2D,
        tensorflow.keras.layers.ConvRecurrent2D,
        tensorflow.keras.layers.Activation,
        tensorflow.keras.layers.ActivityRegularization,
        tensorflow.keras.layers.Dense,
        tensorflow.keras.layers.Dropout,
        tensorflow.keras.layers.Flatten,
        tensorflow.keras.layers.Lambda,
        tensorflow.keras.layers.Masking,
        tensorflow.keras.layers.Permute,
        tensorflow.keras.layers.RepeatVector,
        tensorflow.keras.layers.Reshape,
        tensorflow.keras.layers.SpatialDropout1D,
        tensorflow.keras.layers.SpatialDropout2D,
        tensorflow.keras.layers.SpatialDropout3D,
        tensorflow.keras.layers.CuDNNGRU,
        tensorflow.keras.layers.CuDNNLSTM,
        tensorflow.keras.layers.Embedding,
        tensorflow.keras.layers.LocallyConnected1D,
        tensorflow.keras.layers.LocallyConnected2D,
        tensorflow.keras.layers.Add,
        tensorflow.keras.layers.Average,
        tensorflow.keras.layers.Concatenate,
        tensorflow.keras.layers.Dot,
        tensorflow.keras.layers.Maximum,
        tensorflow.keras.layers.Minimum,
        tensorflow.keras.layers.Multiply,
        tensorflow.keras.layers.Subtract,
        tensorflow.keras.layers.AlphaDropout,
        tensorflow.keras.layers.GaussianDropout,
        tensorflow.keras.layers.GaussianNoise,
        tensorflow.keras.layers.BatchNormalization,
        tensorflow.keras.layers.AveragePooling1D,
        tensorflow.keras.layers.AveragePooling2D,
        tensorflow.keras.layers.AveragePooling3D,
        tensorflow.keras.layers.GlobalAveragePooling1D,
        tensorflow.keras.layers.GlobalAveragePooling2D,
        tensorflow.keras.layers.GlobalAveragePooling3D,
        tensorflow.keras.layers.GlobalMaxPooling1D,
        tensorflow.keras.layers.GlobalMaxPooling2D,
        tensorflow.keras.layers.GlobalMaxPooling3D,
        tensorflow.keras.layers.MaxPooling1D,
        tensorflow.keras.layers.MaxPooling2D,
        tensorflow.keras.layers.MaxPooling3D,
        tensorflow.keras.layers.GRU,
        tensorflow.keras.layers.GRUCell,
        tensorflow.keras.layers.LSTM,
        tensorflow.keras.layers.LSTMCell,
        tensorflow.keras.layers.RNN,
        tensorflow.keras.layers.SimpleRNN,
        tensorflow.keras.layers.SimpleRNNCell,
        tensorflow.keras.layers.StackedRNNCells,
        tensorflow.keras.layers.Bidirectional,
        tensorflow.keras.layers.TimeDistributed,
        tensorflow.keras.layers.Wrapper,
        tensorflow.keras.layers.Highway,
        tensorflow.keras.layers.MaxoutDense,
        tensorflow.keras.layers.Merge,
        tensorflow.keras.layers.Recurrent,
    )
    return KNOWN_LAYERS


def get_activation_search_safe_layers():
    """
    Returns a list of tensorflow.keras layer that we can walk along
    in an activation search.
    """

    # Inside function to not break import if Keras changes.
    ACTIVATION_SEARCH_SAFE_LAYERS = (
        tensorflow.keras.layers.ELU,
        tensorflow.keras.layers.LeakyReLU,
        tensorflow.keras.layers.PReLU,
        tensorflow.keras.layers.Softmax,
        tensorflow.keras.layers.ThresholdedReLU,
        tensorflow.keras.layers.Activation,
        tensorflow.keras.layers.ActivityRegularization,
        tensorflow.keras.layers.Dropout,
        tensorflow.keras.layers.Flatten,
        tensorflow.keras.layers.Reshape,
        tensorflow.keras.layers.Add,
        tensorflow.keras.layers.GaussianNoise,
        tensorflow.keras.layers.BatchNormalization,
    )
    return ACTIVATION_SEARCH_SAFE_LAYERS


###############################################################################
###############################################################################
###############################################################################


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "activation"):
        if activation is not None:
            return layer.activation == tensorflow.keras.activations.get(activation)
        else:
            return True
    elif isinstance(layer, tensorflow.keras.layers.ReLU):
        if activation is not None:
            return (tensorflow.keras.activations.get("relu") ==
                    tensorflow.keras.activations.get(activation))
        else:
            return True
    elif isinstance(layer, (
            tensorflow.keras.layers.ELU,
            tensorflow.keras.layers.LeakyReLU,
            tensorflow.keras.layers.PReLU,
            tensorflow.keras.layers.Softmax,
            tensorflow.keras.layers.ThresholdedReLU)):
        if activation is not None:
            raise Exception("Cannot detect activation type.")
        else:
            return True
    else:
        return False


def contains_kernel(layer):
    """
    Check whether the layer contains a kernel.
    """

    # TODO: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "kernel") or hasattr(layer, "depthwise_kernel") or hasattr(layer, "pointwise_kernel"):
        return True
    else:
        return False


def contains_bias(layer):
    """
    Check whether the layer contains a bias.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "bias"):
        return True
    else:
        return False


def only_relu_activation(layer):
    """Checks if layer contains no or only a ReLU activation."""
    return (not contains_activation(layer) or
            contains_activation(layer, None) or
            contains_activation(layer, "linear") or
            contains_activation(layer, "relu"))


def is_network(layer):
    """
    Is network in network?
    """
    from tcn.tcn import ResidualBlock, TCN
    return isinstance(layer, ResidualBlock) or isinstance(layer, TCN)


def is_conv_layer(layer, *args, **kwargs):
    """Checks if layer is a convolutional layer."""
    CONV_LAYERS = (
        tensorflow.keras.layers.Conv1D,
        tensorflow.keras.layers.Conv2D,
        tensorflow.keras.layers.Conv2DTranspose,
        tensorflow.keras.layers.Conv3D,
        tensorflow.keras.layers.Conv3DTranspose,
        tensorflow.keras.layers.SeparableConv1D,
        tensorflow.keras.layers.SeparableConv2D,
        tensorflow.keras.layers.DepthwiseConv2D
    )
    return isinstance(layer, CONV_LAYERS)


def is_batch_normalization_layer(layer, *args, **kwargs):
    """Checks if layer is a batchnorm layer."""
    return isinstance(layer, tensorflow.keras.layers.BatchNormalization)


def is_add_layer(layer, *args, **kwargs):
    """Checks if layer is an addition-merge layer."""
    return isinstance(layer, tensorflow.keras.layers.Add)


def is_dense_layer(layer, *args, **kwargs):
    """Checks if layer is a dense layer."""
    return isinstance(layer, tensorflow.keras.layers.Dense)


def is_convnet_layer(layer):
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.
    CONVNET_LAYERS = (
        tensorflow.keras.layers.InputLayer,
        tensorflow.keras.layers.ELU,
        tensorflow.keras.layers.LeakyReLU,
        tensorflow.keras.layers.PReLU,
        tensorflow.keras.layers.Softmax,
        tensorflow.keras.layers.ThresholdedReLU,
        tensorflow.keras.layers.Conv1D,
        tensorflow.keras.layers.Conv2D,
        tensorflow.keras.layers.Conv2DTranspose,
        tensorflow.keras.layers.Conv3D,
        tensorflow.keras.layers.Conv3DTranspose,
        tensorflow.keras.layers.Cropping1D,
        tensorflow.keras.layers.Cropping2D,
        tensorflow.keras.layers.Cropping3D,
        tensorflow.keras.layers.SeparableConv1D,
        tensorflow.keras.layers.SeparableConv2D,
        tensorflow.keras.layers.UpSampling1D,
        tensorflow.keras.layers.UpSampling2D,
        tensorflow.keras.layers.UpSampling3D,
        tensorflow.keras.layers.ZeroPadding1D,
        tensorflow.keras.layers.ZeroPadding2D,
        tensorflow.keras.layers.ZeroPadding3D,
        tensorflow.keras.layers.Activation,
        tensorflow.keras.layers.ActivityRegularization,
        tensorflow.keras.layers.Dense,
        tensorflow.keras.layers.Dropout,
        tensorflow.keras.layers.Flatten,
        tensorflow.keras.layers.Lambda,
        tensorflow.keras.layers.Masking,
        tensorflow.keras.layers.Permute,
        tensorflow.keras.layers.RepeatVector,
        tensorflow.keras.layers.Reshape,
        tensorflow.keras.layers.SpatialDropout1D,
        tensorflow.keras.layers.SpatialDropout2D,
        tensorflow.keras.layers.SpatialDropout3D,
        tensorflow.keras.layers.Embedding,
        tensorflow.keras.layers.LocallyConnected1D,
        tensorflow.keras.layers.LocallyConnected2D,
        tensorflow.keras.layers.Add,
        tensorflow.keras.layers.Average,
        tensorflow.keras.layers.Concatenate,
        tensorflow.keras.layers.Dot,
        tensorflow.keras.layers.Maximum,
        tensorflow.keras.layers.Minimum,
        tensorflow.keras.layers.Multiply,
        tensorflow.keras.layers.Subtract,
        tensorflow.keras.layers.AlphaDropout,
        tensorflow.keras.layers.GaussianDropout,
        tensorflow.keras.layers.GaussianNoise,
        tensorflow.keras.layers.BatchNormalization,
        tensorflow.keras.layers.AveragePooling1D,
        tensorflow.keras.layers.AveragePooling2D,
        tensorflow.keras.layers.AveragePooling3D,
        tensorflow.keras.layers.GlobalAveragePooling1D,
        tensorflow.keras.layers.GlobalAveragePooling2D,
        tensorflow.keras.layers.GlobalAveragePooling3D,
        tensorflow.keras.layers.GlobalMaxPooling1D,
        tensorflow.keras.layers.GlobalMaxPooling2D,
        tensorflow.keras.layers.GlobalMaxPooling3D,
        tensorflow.keras.layers.MaxPooling1D,
        tensorflow.keras.layers.MaxPooling2D,
        tensorflow.keras.layers.MaxPooling3D,
    )
    return isinstance(layer, CONVNET_LAYERS)


def is_relu_convnet_layer(layer):
    """Checks if layer is from a convolutional network with ReLUs."""
    return (is_convnet_layer(layer) and only_relu_activation(layer))


def is_average_pooling(layer):
    """Checks if layer is an average-pooling layer."""
    AVERAGEPOOLING_LAYERS = (
        tensorflow.keras.layers.AveragePooling1D,
        tensorflow.keras.layers.AveragePooling2D,
        tensorflow.keras.layers.AveragePooling3D,
        tensorflow.keras.layers.GlobalAveragePooling1D,
        tensorflow.keras.layers.GlobalAveragePooling2D,
        tensorflow.keras.layers.GlobalAveragePooling3D,
    )
    return isinstance(layer, AVERAGEPOOLING_LAYERS)


def is_max_pooling(layer):
    """Checks if layer is a max-pooling layer."""
    MAXPOOLING_LAYERS = (
        tensorflow.keras.layers.MaxPooling1D,
        tensorflow.keras.layers.MaxPooling2D,
        tensorflow.keras.layers.MaxPooling3D,
        tensorflow.keras.layers.GlobalMaxPooling1D,
        tensorflow.keras.layers.GlobalMaxPooling2D,
        tensorflow.keras.layers.GlobalMaxPooling3D,
    )
    return isinstance(layer, MAXPOOLING_LAYERS)


def is_input_layer(layer, ignore_reshape_layers=True):
    """Checks if layer is an input layer."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.
    kgraph = get_kgraph()

    layer_inputs = kgraph.get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        tensorflow.keras.layers.Flatten,
        tensorflow.keras.layers.Permute,
        tensorflow.keras.layers.Reshape,
    )
    while any([isinstance(x, IGNORED_LAYERS) for x in layer_inputs]):
        tmp = set()
        for l in layer_inputs:
            if(ignore_reshape_layers and
               isinstance(l, IGNORED_LAYERS)):
                tmp.update(kgraph.get_input_layers(l))
            else:
                tmp.add(l)
        layer_inputs = tmp

    if all([isinstance(x, tensorflow.keras.layers.InputLayer)
            for x in layer_inputs]):
        return True
    else:
        return False
