# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import inspect
import tensorflow.keras.models as keras_models
import tensorflow.keras.activations as keras_activations
import tensorflow.keras.layers as keras_layers


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
    class_set = set([(getattr(keras_layers, name), name)
                     for name in dir(keras_layers)
                     if (inspect.isclass(getattr(keras_layers, name)) and
                         issubclass(getattr(keras_layers, name),
                                    keras_layers.Layer))])
    return [x[1] for x in sorted((str(x[0]), x[1]) for x in class_set)]


def get_known_layers():
    """
    Returns a list of keras layer we are aware of.
    """

    # Inside function to not break import if Keras changes.
    KNOWN_LAYERS = (
        keras_layers.ELU,
        keras_layers.LeakyReLU,
        keras_layers.PReLU,
        keras_layers.Softmax,
        keras_layers.ThresholdedReLU,
        keras_layers.Conv1D,
        keras_layers.Conv2D,
        keras_layers.Conv2DTranspose,
        keras_layers.Conv3D,
        keras_layers.Conv3DTranspose,
        keras_layers.Cropping1D,
        keras_layers.Cropping2D,
        keras_layers.Cropping3D,
        keras_layers.SeparableConv1D,
        keras_layers.SeparableConv2D,
        keras_layers.UpSampling1D,
        keras_layers.UpSampling2D,
        keras_layers.UpSampling3D,
        keras_layers.ZeroPadding1D,
        keras_layers.ZeroPadding2D,
        keras_layers.ZeroPadding3D,
        keras_layers.ConvLSTM2D,
        keras_layers.ConvRecurrent2D,
        keras_layers.Activation,
        keras_layers.ActivityRegularization,
        keras_layers.Dense,
        keras_layers.Dropout,
        keras_layers.Flatten,
        keras_layers.InputLayer,
        keras_layers.Lambda,
        keras_layers.Masking,
        keras_layers.Permute,
        keras_layers.RepeatVector,
        keras_layers.Reshape,
        keras_layers.SpatialDropout1D,
        keras_layers.SpatialDropout2D,
        keras_layers.SpatialDropout3D,
        keras_layers.CuDNNGRU,
        keras_layers.CuDNNLSTM,
        keras_layers.Embedding,
        keras_layers.LocallyConnected1D,
        keras_layers.LocallyConnected2D,
        keras_layers.Add,
        keras_layers.Average,
        keras_layers.Concatenate,
        keras_layers.Dot,
        keras_layers.Maximum,
        keras_layers.Minimum,
        keras_layers.Multiply,
        keras_layers.Subtract,
        keras_layers.AlphaDropout,
        keras_layers.GaussianDropout,
        keras_layers.GaussianNoise,
        keras_layers.BatchNormalization,
        keras_layers.AveragePooling1D,
        keras_layers.AveragePooling2D,
        keras_layers.AveragePooling3D,
        keras_layers.GlobalAveragePooling1D,
        keras_layers.GlobalAveragePooling2D,
        keras_layers.GlobalAveragePooling3D,
        keras_layers.GlobalMaxPooling1D,
        keras_layers.GlobalMaxPooling2D,
        keras_layers.GlobalMaxPooling3D,
        keras_layers.MaxPooling1D,
        keras_layers.MaxPooling2D,
        keras_layers.MaxPooling3D,
        keras_layers.GRU,
        keras_layers.GRUCell,
        keras_layers.LSTM,
        keras_layers.LSTMCell,
        keras_layers.RNN,
        keras_layers.SimpleRNN,
        keras_layers.SimpleRNNCell,
        keras_layers.StackedRNNCells,
        keras_layers.Bidirectional,
        keras_layers.TimeDistributed,
        keras_layers.Wrapper,
    )
    return KNOWN_LAYERS


def get_activation_search_safe_layers():
    """
    Returns a list of keras layer that we can walk along
    in an activation search.
    """

    # Inside function to not break import if Keras changes.
    ACTIVATION_SEARCH_SAFE_LAYERS = (
        keras_layers.ELU,
        keras_layers.LeakyReLU,
        keras_layers.PReLU,
        keras_layers.Softmax,
        keras_layers.ThresholdedReLU,
        keras_layers.Activation,
        keras_layers.ActivityRegularization,
        keras_layers.Dropout,
        keras_layers.Flatten,
        keras_layers.Reshape,
        keras_layers.Add,
        keras_layers.GaussianNoise,
        keras_layers.BatchNormalization,
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
            return layer.activation == keras_activations.get(activation)
        else:
            return True
    elif isinstance(layer, keras_layers.ReLU):
        if activation is not None:
            return (keras_activations.get("relu") ==
                    keras_activations.get(activation))
        else:
            return True
    elif isinstance(layer, (
            keras_layers.ELU,
            keras_layers.LeakyReLU,
            keras_layers.PReLU,
            keras_layers.Softmax,
            keras_layers.ThresholdedReLU)):
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
    return isinstance(layer, keras_models.Model)


def is_conv_layer(layer, *args, **kwargs):
    """Checks if layer is a convolutional layer."""
    CONV_LAYERS = (
        keras_layers.Conv1D,
        keras_layers.Conv2D,
        keras_layers.Conv2DTranspose,
        keras_layers.Conv3D,
        keras_layers.Conv3DTranspose,
        keras_layers.SeparableConv1D,
        keras_layers.SeparableConv2D,
        keras_layers.DepthwiseConv2D
    )
    return isinstance(layer, CONV_LAYERS)


def is_batch_normalization_layer(layer, *args, **kwargs):
    """Checks if layer is a batchnorm layer."""
    return isinstance(layer, keras_layers.BatchNormalization)


def is_add_layer(layer, *args, **kwargs):
    """Checks if layer is an addition-merge layer."""
    return isinstance(layer, keras_layers.Add)


def is_dense_layer(layer, *args, **kwargs):
    """Checks if layer is a dense layer."""
    return isinstance(layer, keras_layers.Dense)


def is_convnet_layer(layer):
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.
    CONVNET_LAYERS = (
        keras_layers.ELU,
        keras_layers.LeakyReLU,
        keras_layers.PReLU,
        keras_layers.Softmax,
        keras_layers.ThresholdedReLU,
        keras_layers.Conv1D,
        keras_layers.Conv2D,
        keras_layers.Conv2DTranspose,
        keras_layers.Conv3D,
        keras_layers.Conv3DTranspose,
        keras_layers.Cropping1D,
        keras_layers.Cropping2D,
        keras_layers.Cropping3D,
        keras_layers.SeparableConv1D,
        keras_layers.SeparableConv2D,
        keras_layers.UpSampling1D,
        keras_layers.UpSampling2D,
        keras_layers.UpSampling3D,
        keras_layers.ZeroPadding1D,
        keras_layers.ZeroPadding2D,
        keras_layers.ZeroPadding3D,
        keras_layers.Activation,
        keras_layers.ActivityRegularization,
        keras_layers.Dense,
        keras_layers.Dropout,
        keras_layers.Flatten,
        keras_layers.InputLayer,
        keras_layers.Lambda,
        keras_layers.Masking,
        keras_layers.Permute,
        keras_layers.RepeatVector,
        keras_layers.Reshape,
        keras_layers.SpatialDropout1D,
        keras_layers.SpatialDropout2D,
        keras_layers.SpatialDropout3D,
        keras_layers.Embedding,
        keras_layers.LocallyConnected1D,
        keras_layers.LocallyConnected2D,
        keras_layers.Add,
        keras_layers.Average,
        keras_layers.Concatenate,
        keras_layers.Dot,
        keras_layers.Maximum,
        keras_layers.Minimum,
        keras_layers.Multiply,
        keras_layers.Subtract,
        keras_layers.AlphaDropout,
        keras_layers.GaussianDropout,
        keras_layers.GaussianNoise,
        keras_layers.BatchNormalization,
        keras_layers.AveragePooling1D,
        keras_layers.AveragePooling2D,
        keras_layers.AveragePooling3D,
        keras_layers.GlobalAveragePooling1D,
        keras_layers.GlobalAveragePooling2D,
        keras_layers.GlobalAveragePooling3D,
        keras_layers.GlobalMaxPooling1D,
        keras_layers.GlobalMaxPooling2D,
        keras_layers.GlobalMaxPooling3D,
        keras_layers.MaxPooling1D,
        keras_layers.MaxPooling2D,
        keras_layers.MaxPooling3D,
    )
    return isinstance(layer, CONVNET_LAYERS)


def is_relu_convnet_layer(layer):
    """Checks if layer is from a convolutional network with ReLUs."""
    return (is_convnet_layer(layer) and only_relu_activation(layer))


def is_average_pooling(layer):
    """Checks if layer is an average-pooling layer."""
    AVERAGEPOOLING_LAYERS = (
        keras_layers.AveragePooling1D,
        keras_layers.AveragePooling2D,
        keras_layers.AveragePooling3D,
        keras_layers.GlobalAveragePooling1D,
        keras_layers.GlobalAveragePooling2D,
        keras_layers.GlobalAveragePooling3D,
    )
    return isinstance(layer, AVERAGEPOOLING_LAYERS)


def is_max_pooling(layer):
    """Checks if layer is a max-pooling layer."""
    MAXPOOLING_LAYERS = (
        keras_layers.MaxPooling1D,
        keras_layers.MaxPooling2D,
        keras_layers.MaxPooling3D,
        keras_layers.GlobalMaxPooling1D,
        keras_layers.GlobalMaxPooling2D,
        keras_layers.GlobalMaxPooling3D,
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
        keras_layers.Flatten,
        keras_layers.Permute,
        keras_layers.Reshape,
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

    if all([isinstance(x, keras_layers.InputLayer)
            for x in layer_inputs]):
        return True
    else:
        return False


def is_layer_at_idx(layer, index, ignore_reshape_layers=True):
    """Checks if layer is a layer at index index."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.
    kgraph = get_kgraph()

    layer_inputs = [layer]
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        keras_layers.Flatten,
        keras_layers.Permute,
        keras_layers.Reshape,
    )

    for i in range(index):

        while any([isinstance(x, IGNORED_LAYERS) for x in layer_inputs]):
            tmp = set()
            for l in layer_inputs:
                if (ignore_reshape_layers and
                        isinstance(l, IGNORED_LAYERS)):
                    tmp.update(kgraph.get_input_layers(l))
                else:
                    tmp.add(l)
            layer_inputs = tmp

        tmp = set()
        for l in layer_inputs:
            tmp.update(kgraph.get_input_layers(l))
        layer_inputs = tmp

        if any([isinstance(x, keras_layers.InputLayer)
                for x in layer_inputs]):
            return False

    ret = all([is_input_layer(x, ignore_reshape_layers) for x in layer_inputs])
    return ret
