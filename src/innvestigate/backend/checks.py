"""Check Keras Layers for properties,
e.g. if it is an input or a pooling layer"""

from __future__ import annotations

import tensorflow.keras.layers as klayers
from tensorflow import Module, keras

import innvestigate.backend as ibackend
from innvestigate.backend.types import Layer

__all__ = [
    "get_activation_search_safe_layers",
    "contains_activation",
    "contains_kernel",
    "only_relu_activation",
    "is_module",
    "is_convnet_layer",
    "is_average_pooling",
    "is_max_pooling",
    "is_input_layer",
    "is_batch_normalization_layer",
    "is_embedding_layer",
]


def get_activation_search_safe_layers():
    """
    Returns a list of keras layer that we can walk along
    in an activation search.
    """

    # Inside function to not break import if Keras changes.
    activation_search_safe_layers = (
        klayers.ELU,
        klayers.LeakyReLU,
        klayers.PReLU,
        klayers.Softmax,
        klayers.ThresholdedReLU,
        klayers.Activation,
        klayers.ActivityRegularization,
        klayers.Dropout,
        klayers.Flatten,
        klayers.Reshape,
        klayers.Add,
        klayers.GaussianNoise,
        klayers.BatchNormalization,
    )
    return activation_search_safe_layers


###############################################################################


def contains_activation(layer: Layer, activation: str = None) -> bool:
    """Check whether the layer contains an activation function of type `activation`.
    If `activation` is None, only check if layer can contain an activation.

    :param layer: Keras layer to check
    :type layer: Layer
    :param activation: Keras name of activation function, defaults to None
    :type activation: str, optional
    :return: If `activation` is None, check if layer contains any activation function.
        Otherwise check for specific activation function of type `activation`.
    :rtype: bool
    """

    if activation is None:
        return contains_any_activation(layer)
    a = activation.lower()

    # Check for activations in layer
    if hasattr(layer, "activation"):
        return bool(layer.activation == keras.activations.get(a))

    # Check for layers that are activations
    if a == "softmax":
        return isinstance(layer, (klayers.Softmax))
    if a == "relu":
        return isinstance(layer, (klayers.ReLU))
    if a == "elu":
        return isinstance(layer, (klayers.ELU))
    if a == "prelu":
        return isinstance(layer, (klayers.PReLU))
    if a == "leakyrelu":
        return isinstance(layer, (klayers.LeakyReLU))
    if a == "thresholdedrelu":
        return isinstance(layer, (klayers.ThresholdedReLU))
    return False


ACTIVATION_LAYERS = (
    klayers.ReLU,
    klayers.ELU,
    klayers.LeakyReLU,
    klayers.PReLU,
    klayers.Softmax,
    klayers.ThresholdedReLU,
)


def contains_any_activation(layer: Layer) -> bool:
    """Check whether layer contains any activation or is activation layer.

    :param layer: Keras layer to check
    :type layer: Layer
    :return: True if activation was found.
    :rtype: bool
    """

    return hasattr(layer, "activation") or isinstance(layer, ACTIVATION_LAYERS)


def contains_kernel(layer: Layer) -> bool:
    """
    Check whether the layer contains a kernel.
    """

    # TODO: add test and check this more throughroughly.
    # rely on Keras convention.
    return (
        hasattr(layer, "kernel")
        or hasattr(layer, "depthwise_kernel")
        or hasattr(layer, "pointwise_kernel")
    )


def only_relu_activation(layer: Layer) -> bool:
    """Checks if layer contains no or only a ReLU activation."""
    return (
        not contains_activation(layer)
        or contains_activation(layer, "linear")
        or contains_activation(layer, "relu")
    )


def is_module(layer) -> bool:
    """
    Is there a tf.Module in the network?
    """
    if isinstance(layer, klayers.Layer):
        return False
    return isinstance(layer, Module)


CONV_LAYERS = (
    klayers.Conv1D,
    klayers.Conv2D,
    klayers.Conv2DTranspose,
    klayers.Conv3D,
    klayers.Conv3DTranspose,
    klayers.SeparableConv1D,
    klayers.SeparableConv2D,
    klayers.DepthwiseConv2D,
)


def is_conv_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is a convolutional layer."""

    return isinstance(layer, CONV_LAYERS)


def is_embedding_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is an embedding layer."""
    return isinstance(layer, klayers.Embedding)


def is_batch_normalization_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is a batchnorm layer."""
    return isinstance(layer, klayers.BatchNormalization)


def is_add_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is an addition-merge layer."""
    return isinstance(layer, klayers.Add)


def is_dense_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is a dense layer."""
    return isinstance(layer, klayers.Dense)


CONVNET_LAYERS = (
    klayers.InputLayer,
    klayers.ELU,
    klayers.LeakyReLU,
    klayers.PReLU,
    klayers.Softmax,
    klayers.ThresholdedReLU,
    klayers.Conv1D,
    klayers.Conv2D,
    klayers.Conv2DTranspose,
    klayers.Conv3D,
    klayers.Conv3DTranspose,
    klayers.Cropping1D,
    klayers.Cropping2D,
    klayers.Cropping3D,
    klayers.SeparableConv1D,
    klayers.SeparableConv2D,
    klayers.UpSampling1D,
    klayers.UpSampling2D,
    klayers.UpSampling3D,
    klayers.ZeroPadding1D,
    klayers.ZeroPadding2D,
    klayers.ZeroPadding3D,
    klayers.Activation,
    klayers.ActivityRegularization,
    klayers.Dense,
    klayers.Dropout,
    klayers.Flatten,
    klayers.Lambda,
    klayers.Masking,
    klayers.Permute,
    klayers.RepeatVector,
    klayers.Reshape,
    klayers.SpatialDropout1D,
    klayers.SpatialDropout2D,
    klayers.SpatialDropout3D,
    klayers.Embedding,
    klayers.LocallyConnected1D,
    klayers.LocallyConnected2D,
    klayers.Add,
    klayers.Average,
    klayers.Concatenate,
    klayers.Dot,
    klayers.Maximum,
    klayers.Minimum,
    klayers.Multiply,
    klayers.Subtract,
    klayers.AlphaDropout,
    klayers.GaussianDropout,
    klayers.GaussianNoise,
    klayers.BatchNormalization,
    klayers.AveragePooling1D,
    klayers.AveragePooling2D,
    klayers.AveragePooling3D,
    klayers.GlobalAveragePooling1D,
    klayers.GlobalAveragePooling2D,
    klayers.GlobalAveragePooling3D,
    klayers.GlobalMaxPooling1D,
    klayers.GlobalMaxPooling2D,
    klayers.GlobalMaxPooling3D,
    klayers.MaxPooling1D,
    klayers.MaxPooling2D,
    klayers.MaxPooling3D,
)


def is_convnet_layer(layer: Layer) -> bool:
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.

    return isinstance(layer, CONVNET_LAYERS)


AVERAGEPOOLING_LAYERS = (
    klayers.AveragePooling1D,
    klayers.AveragePooling2D,
    klayers.AveragePooling3D,
    klayers.GlobalAveragePooling1D,
    klayers.GlobalAveragePooling2D,
    klayers.GlobalAveragePooling3D,
)


def is_average_pooling(layer: Layer) -> bool:
    """Checks if layer is an average-pooling layer."""

    return isinstance(layer, AVERAGEPOOLING_LAYERS)


MAXPOOLING_LAYERS = (
    klayers.MaxPooling1D,
    klayers.MaxPooling2D,
    klayers.MaxPooling3D,
    klayers.GlobalMaxPooling1D,
    klayers.GlobalMaxPooling2D,
    klayers.GlobalMaxPooling3D,
)


def is_max_pooling(layer: Layer) -> bool:
    """Checks if layer is a max-pooling layer."""
    return isinstance(layer, MAXPOOLING_LAYERS)


def get_input_layers(layer: Layer) -> set[Layer]:
    """Returns all layers that created this layer's inputs."""
    ret = set()

    for node_index in range(len(layer._inbound_nodes)):
        Xs = ibackend.to_list(layer.get_input_at(node_index))
        for X in Xs:
            ret.add(X._keras_history[0])

    return ret


def is_input_layer(layer: Layer, ignore_reshape_layers: bool = True) -> bool:
    """Checks if layer is an input layer."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.

    layer_inputs = get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # TODO: update this list!
    ignored_layers = (
        klayers.Flatten,
        klayers.Permute,
        klayers.Reshape,
    )
    while any(isinstance(x, ignored_layers) for x in layer_inputs):
        tmp = set()
        for l in layer_inputs:
            if ignore_reshape_layers and isinstance(l, ignored_layers):
                tmp.update(get_input_layers(l))
            else:
                tmp.add(l)
        layer_inputs = tmp

    return all(isinstance(x, klayers.InputLayer) for x in layer_inputs)


def is_layer_at_idx(layer: Layer, index, ignore_reshape_layers=True) -> bool:
    """Checks if layer is a layer at index index,
    by repeatedly applying is_input_layer()."""
    # TODO: implement layer index check
    raise NotImplementedError("Layer index checking hasn't been implemented yet.")
