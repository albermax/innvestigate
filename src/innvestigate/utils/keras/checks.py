"""Check Keras Layers for properties,
e.g. if it is an input or a pooling layer"""

from __future__ import annotations

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

import innvestigate.utils.keras.graph as kgraph
from innvestigate.utils.types import Layer

__all__ = [
    "get_activation_search_safe_layers",
    "contains_activation",
    "contains_kernel",
    "only_relu_activation",
    "is_network",
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

    if activation is not None:
        if hasattr(layer, "activation"):
            return bool(layer.activation == keras.activations.get(activation))
        elif (activation == "relu") and isinstance(
            layer,
            (
                keras.layers.ReLU,
                keras.layers.advanced_activations.ReLU,
            ),
        ):
            return True
        elif isinstance(
            layer,
            (
                keras.layers.advanced_activations.ELU,
                keras.layers.advanced_activations.LeakyReLU,
                keras.layers.advanced_activations.PReLU,
                keras.layers.advanced_activations.Softmax,
                keras.layers.advanced_activations.ThresholdedReLU,
            ),
        ):
            raise Exception(f"Cannot detect activation type, expected {activation}.")
    else:  # just check if layer contains activation
        if hasattr(layer, "activation") or isinstance(
            layer,
            (
                keras.layers.advanced_activations.ELU,
                keras.layers.advanced_activations.LeakyReLU,
                keras.layers.advanced_activations.PReLU,
                keras.layers.advanced_activations.Softmax,
                keras.layers.advanced_activations.ThresholdedReLU,
            ),
        ):
            return True
    return False


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
        or contains_activation(layer, None)
        or contains_activation(layer, "linear")
        or contains_activation(layer, "relu")
    )


def is_network(layer: Layer) -> bool:
    """
    Is network in network?
    """
    return isinstance(layer, keras.engine.topology.Network)


def is_conv_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is a convolutional layer."""
    conv_layers = (
        keras.layers.convolutional.Conv1D,
        keras.layers.convolutional.Conv2D,
        keras.layers.convolutional.Conv2DTranspose,
        keras.layers.convolutional.Conv3D,
        keras.layers.convolutional.Conv3DTranspose,
        keras.layers.convolutional.SeparableConv1D,
        keras.layers.convolutional.SeparableConv2D,
        keras.layers.convolutional.DepthwiseConv2D,
    )
    return isinstance(layer, conv_layers)


def is_embedding_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is an embedding layer."""
    return isinstance(layer, keras.layers.Embedding)


def is_batch_normalization_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is a batchnorm layer."""
    return isinstance(layer, keras.layers.normalization.BatchNormalization)


def is_add_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is an addition-merge layer."""
    return isinstance(layer, keras.layers.Add)


def is_dense_layer(layer: Layer, *_args, **_kwargs) -> bool:
    """Checks if layer is a dense layer."""
    return isinstance(layer, keras.layers.core.Dense)


def is_convnet_layer(layer: Layer) -> bool:
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.
    convnet_layers = (
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
    return isinstance(layer, convnet_layers)


def is_average_pooling(layer: Layer) -> bool:
    """Checks if layer is an average-pooling layer."""
    averagepooling_layers = (
        keras.layers.pooling.AveragePooling1D,
        keras.layers.pooling.AveragePooling2D,
        keras.layers.pooling.AveragePooling3D,
        keras.layers.pooling.GlobalAveragePooling1D,
        keras.layers.pooling.GlobalAveragePooling2D,
        keras.layers.pooling.GlobalAveragePooling3D,
    )
    return isinstance(layer, averagepooling_layers)


def is_max_pooling(layer: Layer) -> bool:
    """Checks if layer is a max-pooling layer."""
    maxpooling_layers = (
        keras.layers.pooling.MaxPooling1D,
        keras.layers.pooling.MaxPooling2D,
        keras.layers.pooling.MaxPooling3D,
        keras.layers.pooling.GlobalMaxPooling1D,
        keras.layers.pooling.GlobalMaxPooling2D,
        keras.layers.pooling.GlobalMaxPooling3D,
    )
    return isinstance(layer, maxpooling_layers)


def is_input_layer(layer: Layer, ignore_reshape_layers: bool = True) -> bool:
    """Checks if layer is an input layer."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.

    layer_inputs = kgraph.get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # TODO: update this list!
    ignored_layers = (
        keras.layers.Flatten,
        keras.layers.Permute,
        keras.layers.Reshape,
    )
    while any(isinstance(x, ignored_layers) for x in layer_inputs):
        tmp = set()
        for layer_input in layer_inputs:
            if ignore_reshape_layers and isinstance(layer_input, ignored_layers):
                tmp.update(kgraph.get_input_layers(layer_input))
            else:
                tmp.add(layer_input)
        layer_inputs = tmp

    return all(isinstance(x, keras.layers.InputLayer) for x in layer_inputs)


def is_layer_at_idx(layer: Layer, index, ignore_reshape_layers=True) -> bool:
    """Checks if layer is a layer at index index,
    by repeatedly applying is_input_layer()."""
    # TODO: implement layer index check
    raise NotImplementedError("Layer index checking hasn't been implemented yet.")
