# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import inspect
import keras.engine.topology
import keras.layers
import keras.layers.advanced_activations
import keras.layers.convolutional
import keras.layers.convolutional_recurrent
import keras.layers.core
import keras.layers.cudnn_recurrent
import keras.layers.embeddings
import keras.layers.local
import keras.layers.noise
import keras.layers.normalization
import keras.layers.pooling
import keras.layers.recurrent
import keras.layers.wrappers
import keras.legacy.layers


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
    class_set = set([(getattr(keras.layers, name), name)
                     for name in dir(keras.layers)
                     if (inspect.isclass(getattr(keras.layers, name)) and
                         issubclass(getattr(keras.layers, name),
                                    keras.engine.topology.Layer))])
    return [x[1] for x in sorted((str(x[0]), x[1]) for x in class_set)]


def get_known_layers():
    """
    Returns a list of keras layer we are aware of.
    """

    # Inside function to not break import if Keras changes.
    KNOWN_LAYERS = (
        keras.engine.topology.InputLayer,
        keras.layers.advanced_activations.ELU,
        keras.layers.advanced_activations.LeakyReLU,
        keras.layers.advanced_activations.PReLU,
        keras.layers.advanced_activations.Softmax,
        keras.layers.advanced_activations.ThresholdedReLU,
        keras.layers.convolutional.Conv1D,
        keras.layers.convolutional.Conv2D,
        keras.layers.convolutional.Conv2DTranspose,
        keras.layers.convolutional.Conv3D,
        keras.layers.convolutional.Conv3DTranspose,
        keras.layers.convolutional.Cropping1D,
        keras.layers.convolutional.Cropping2D,
        keras.layers.convolutional.Cropping3D,
        keras.layers.convolutional.SeparableConv1D,
        keras.layers.convolutional.SeparableConv2D,
        keras.layers.convolutional.UpSampling1D,
        keras.layers.convolutional.UpSampling2D,
        keras.layers.convolutional.UpSampling3D,
        keras.layers.convolutional.ZeroPadding1D,
        keras.layers.convolutional.ZeroPadding2D,
        keras.layers.convolutional.ZeroPadding3D,
        keras.layers.convolutional_recurrent.ConvLSTM2D,
        keras.layers.convolutional_recurrent.ConvRecurrent2D,
        keras.layers.core.Activation,
        keras.layers.core.ActivityRegularization,
        keras.layers.core.Dense,
        keras.layers.core.Dropout,
        keras.layers.core.Flatten,
        keras.layers.core.Lambda,
        keras.layers.core.Masking,
        keras.layers.core.Permute,
        keras.layers.core.RepeatVector,
        keras.layers.core.Reshape,
        keras.layers.core.SpatialDropout1D,
        keras.layers.core.SpatialDropout2D,
        keras.layers.core.SpatialDropout3D,
        keras.layers.cudnn_recurrent.CuDNNGRU,
        keras.layers.cudnn_recurrent.CuDNNLSTM,
        keras.layers.embeddings.Embedding,
        keras.layers.local.LocallyConnected1D,
        keras.layers.local.LocallyConnected2D,
        keras.layers.Add,
        keras.layers.Average,
        keras.layers.Concatenate,
        keras.layers.Dot,
        keras.layers.Maximum,
        keras.layers.Minimum,
        keras.layers.Multiply,
        keras.layers.Subtract,
        keras.layers.noise.AlphaDropout,
        keras.layers.noise.GaussianDropout,
        keras.layers.noise.GaussianNoise,
        keras.layers.normalization.BatchNormalization,
        keras.layers.pooling.AveragePooling1D,
        keras.layers.pooling.AveragePooling2D,
        keras.layers.pooling.AveragePooling3D,
        keras.layers.pooling.GlobalAveragePooling1D,
        keras.layers.pooling.GlobalAveragePooling2D,
        keras.layers.pooling.GlobalAveragePooling3D,
        keras.layers.pooling.GlobalMaxPooling1D,
        keras.layers.pooling.GlobalMaxPooling2D,
        keras.layers.pooling.GlobalMaxPooling3D,
        keras.layers.pooling.MaxPooling1D,
        keras.layers.pooling.MaxPooling2D,
        keras.layers.pooling.MaxPooling3D,
        keras.layers.recurrent.GRU,
        keras.layers.recurrent.GRUCell,
        keras.layers.recurrent.LSTM,
        keras.layers.recurrent.LSTMCell,
        keras.layers.recurrent.RNN,
        keras.layers.recurrent.SimpleRNN,
        keras.layers.recurrent.SimpleRNNCell,
        keras.layers.recurrent.StackedRNNCells,
        keras.layers.wrappers.Bidirectional,
        keras.layers.wrappers.TimeDistributed,
        keras.layers.wrappers.Wrapper,
        keras.legacy.layers.Highway,
        keras.legacy.layers.MaxoutDense,
        keras.legacy.layers.Merge,
        keras.legacy.layers.Recurrent,
    )
    return KNOWN_LAYERS


def get_activation_search_safe_layers():
    """
    Returns a list of keras layer that we can walk along
    in an activation search.
    """

    # Inside function to not break import if Keras changes.
    ACTIVATION_SEARCH_SAFE_LAYERS = (
        keras.layers.advanced_activations.ELU,
        keras.layers.advanced_activations.LeakyReLU,
        keras.layers.advanced_activations.PReLU,
        keras.layers.advanced_activations.Softmax,
        keras.layers.advanced_activations.ThresholdedReLU,
        keras.layers.core.Activation,
        keras.layers.core.ActivityRegularization,
        keras.layers.core.Dropout,
        keras.layers.core.Flatten,
        keras.layers.core.Reshape,
        keras.layers.Add,
        keras.layers.noise.GaussianNoise,
        keras.layers.normalization.BatchNormalization,
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
            return layer.activation == keras.activations.get(activation)
        else:
            return True
    elif isinstance(layer, keras.layers.ReLU):
        if activation is not None:
            return (keras.activations.get("relu") ==
                    keras.activations.get(activation))
        else:
            return True
    elif isinstance(layer, (
            keras.layers.advanced_activations.ELU,
            keras.layers.advanced_activations.LeakyReLU,
            keras.layers.advanced_activations.PReLU,
            keras.layers.advanced_activations.Softmax,
            keras.layers.advanced_activations.ThresholdedReLU)):
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
    return isinstance(layer, keras.engine.topology.Network)


def is_conv_layer(layer, *args, **kwargs):
    """Checks if layer is a convolutional layer."""
    CONV_LAYERS = (
        keras.layers.convolutional.Conv1D,
        keras.layers.convolutional.Conv2D,
        keras.layers.convolutional.Conv2DTranspose,
        keras.layers.convolutional.Conv3D,
        keras.layers.convolutional.Conv3DTranspose,
        keras.layers.convolutional.SeparableConv1D,
        keras.layers.convolutional.SeparableConv2D,
        keras.layers.convolutional.DepthwiseConv2D
    )
    return isinstance(layer, CONV_LAYERS)


def is_batch_normalization_layer(layer, *args, **kwargs):
    """Checks if layer is a batchnorm layer."""
    return isinstance(layer, keras.layers.normalization.BatchNormalization)


def is_add_layer(layer, *args, **kwargs):
    """Checks if layer is an addition-merge layer."""
    return isinstance(layer, keras.layers.Add)


def is_dense_layer(layer, *args, **kwargs):
    """Checks if layer is a dense layer."""
    return isinstance(layer, keras.layers.core.Dense)


def is_convnet_layer(layer):
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.
    CONVNET_LAYERS = (
        keras.engine.topology.InputLayer,
        keras.layers.advanced_activations.ELU,
        keras.layers.advanced_activations.LeakyReLU,
        keras.layers.advanced_activations.PReLU,
        keras.layers.advanced_activations.Softmax,
        keras.layers.advanced_activations.ThresholdedReLU,
        keras.layers.convolutional.Conv1D,
        keras.layers.convolutional.Conv2D,
        keras.layers.convolutional.Conv2DTranspose,
        keras.layers.convolutional.Conv3D,
        keras.layers.convolutional.Conv3DTranspose,
        keras.layers.convolutional.Cropping1D,
        keras.layers.convolutional.Cropping2D,
        keras.layers.convolutional.Cropping3D,
        keras.layers.convolutional.SeparableConv1D,
        keras.layers.convolutional.SeparableConv2D,
        keras.layers.convolutional.UpSampling1D,
        keras.layers.convolutional.UpSampling2D,
        keras.layers.convolutional.UpSampling3D,
        keras.layers.convolutional.ZeroPadding1D,
        keras.layers.convolutional.ZeroPadding2D,
        keras.layers.convolutional.ZeroPadding3D,
        keras.layers.core.Activation,
        keras.layers.core.ActivityRegularization,
        keras.layers.core.Dense,
        keras.layers.core.Dropout,
        keras.layers.core.Flatten,
        keras.layers.core.Lambda,
        keras.layers.core.Masking,
        keras.layers.core.Permute,
        keras.layers.core.RepeatVector,
        keras.layers.core.Reshape,
        keras.layers.core.SpatialDropout1D,
        keras.layers.core.SpatialDropout2D,
        keras.layers.core.SpatialDropout3D,
        keras.layers.embeddings.Embedding,
        keras.layers.local.LocallyConnected1D,
        keras.layers.local.LocallyConnected2D,
        keras.layers.Add,
        keras.layers.Average,
        keras.layers.Concatenate,
        keras.layers.Dot,
        keras.layers.Maximum,
        keras.layers.Minimum,
        keras.layers.Multiply,
        keras.layers.Subtract,
        keras.layers.noise.AlphaDropout,
        keras.layers.noise.GaussianDropout,
        keras.layers.noise.GaussianNoise,
        keras.layers.normalization.BatchNormalization,
        keras.layers.pooling.AveragePooling1D,
        keras.layers.pooling.AveragePooling2D,
        keras.layers.pooling.AveragePooling3D,
        keras.layers.pooling.GlobalAveragePooling1D,
        keras.layers.pooling.GlobalAveragePooling2D,
        keras.layers.pooling.GlobalAveragePooling3D,
        keras.layers.pooling.GlobalMaxPooling1D,
        keras.layers.pooling.GlobalMaxPooling2D,
        keras.layers.pooling.GlobalMaxPooling3D,
        keras.layers.pooling.MaxPooling1D,
        keras.layers.pooling.MaxPooling2D,
        keras.layers.pooling.MaxPooling3D,
    )
    return isinstance(layer, CONVNET_LAYERS)


def is_relu_convnet_layer(layer):
    """Checks if layer is from a convolutional network with ReLUs."""
    return (is_convnet_layer(layer) and only_relu_activation(layer))


def is_average_pooling(layer):
    """Checks if layer is an average-pooling layer."""
    AVERAGEPOOLING_LAYERS = (
        keras.layers.pooling.AveragePooling1D,
        keras.layers.pooling.AveragePooling2D,
        keras.layers.pooling.AveragePooling3D,
        keras.layers.pooling.GlobalAveragePooling1D,
        keras.layers.pooling.GlobalAveragePooling2D,
        keras.layers.pooling.GlobalAveragePooling3D,
    )
    return isinstance(layer, AVERAGEPOOLING_LAYERS)


def is_max_pooling(layer):
    """Checks if layer is a max-pooling layer."""
    MAXPOOLING_LAYERS = (
        keras.layers.pooling.MaxPooling1D,
        keras.layers.pooling.MaxPooling2D,
        keras.layers.pooling.MaxPooling3D,
        keras.layers.pooling.GlobalMaxPooling1D,
        keras.layers.pooling.GlobalMaxPooling2D,
        keras.layers.pooling.GlobalMaxPooling3D,
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
        keras.layers.Flatten,
        keras.layers.Permute,
        keras.layers.Reshape,
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

    if all([isinstance(x, keras.layers.InputLayer)
            for x in layer_inputs]):
        return True
    else:
        return False
