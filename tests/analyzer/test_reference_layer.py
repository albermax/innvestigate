"""
Test all LRP Analyzers over single layer models using random weights and random input.
"""
import os

import h5py
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras

from innvestigate.analyzer.deeptaylor import BoundedDeepTaylor, DeepTaylor
from innvestigate.analyzer.gradient_based import (
    Deconvnet,
    Gradient,
    GuidedBackprop,
    InputTimesGradient,
    IntegratedGradients,
    SmoothGrad,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPZ,
    LRPAlpha1Beta0,
    LRPAlpha1Beta0IgnoreBias,
    LRPAlpha2Beta1,
    LRPAlpha2Beta1IgnoreBias,
    LRPEpsilon,
    LRPEpsilonIgnoreBias,
    LRPFlat,
    LRPSequentialPresetA,
    LRPSequentialPresetAFlat,
    LRPSequentialPresetB,
    LRPSequentialPresetBFlat,
    LRPWSquare,
    LRPZIgnoreBias,
    LRPZPlus,
    LRPZPlusFast,
)

tf.compat.v1.disable_eager_execution()

methods = {
    # Gradient based
    "Gradient": (Gradient, {}),
    "InputTimesGradient": (InputTimesGradient, {}),
    "Deconvnet": (Deconvnet, {}),
    "GuidedBackprop": (GuidedBackprop, {}),
    "IntegratedGradients": (IntegratedGradients, {}),
    "SmoothGrad": (SmoothGrad, {}),
    # Relevance based
    "LRPZ": (LRPZ, {}),
    "LRPZ_Flat_input_layer_rule": (LRPZ, {"input_layer_rule": "Flat"}),
    "LRPZ_boxed_input_layer_rule": (LRPZ, {"input_layer_rule": (-10, 10)}),
    "LRPZIgnoreBias": (LRPZIgnoreBias, {}),
    "LRPZPlus": (LRPZPlus, {}),
    "LRPZPlusFast": (LRPZPlusFast, {}),
    "LRPAlpha1Beta0": (LRPAlpha1Beta0, {}),
    "LRPAlpha1Beta0IgnoreBias": (LRPAlpha1Beta0IgnoreBias, {}),
    "LRPAlpha2Beta1": (LRPAlpha2Beta1, {}),
    "LRPAlpha2Beta1IgnoreBias": (LRPAlpha2Beta1IgnoreBias, {}),
    "LRPEpsilon": (LRPEpsilon, {}),
    "LRPEpsilonIgnoreBias": (LRPEpsilonIgnoreBias, {}),
    "LRPFlat": (LRPFlat, {}),
    "LRPWSquare": (LRPWSquare, {}),
    "LRPSequentialPresetA": (LRPSequentialPresetA, {}),
    "LRPSequentialPresetB": (LRPSequentialPresetB, {}),
    "LRPSequentialPresetAFlat": (LRPSequentialPresetAFlat, {}),
    "LRPSequentialPresetBFlat": (LRPSequentialPresetBFlat, {}),
    # Deep taylor
    "DeepTaylor": (DeepTaylor, {}),
    "BoundedDeepTaylor": (BoundedDeepTaylor, {"low": -128, "high": 128}),
}

rtol = 1e-3
atol = 1e-5

# Sizes used for data generation
input_shape = (10, 10, 3)
batch_size = 1
kernel_size = (3, 3)
pool_size = (2, 2)

LAYERS_2D = {
    "Dense": keras.layers.Dense(5, input_shape=input_shape),
    "Dense_relu": keras.layers.Dense(5, activation="relu", input_shape=input_shape),
    "Conv2D": keras.layers.Conv2D(5, kernel_size, input_shape=input_shape),
    "Conv2D_relu": keras.layers.Conv2D(
        5, kernel_size, activation="relu", input_shape=input_shape
    ),
    "AveragePooling2D": keras.layers.AveragePooling2D(
        pool_size, input_shape=input_shape
    ),
    "MaxPooling2D": keras.layers.MaxPooling2D(pool_size, input_shape=input_shape),
}


def debug_failed_all_close(val, ref, val_name, layer_name, analyzer_name):
    diff = np.absolute(val - ref)
    # Function evaluated by np.allclose, see "Notes":
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    tol = atol + rtol * np.absolute(ref)
    idx = np.argwhere(diff > tol)

    print(
        f"{len(idx)}/{np.prod(val.shape)} "
        f'failed on referece "{val_name}" using layer {layer_name} with {analyzer_name}'
        f"(atol={atol}, rtol={rtol})"
    )
    for i in idx:
        ti = tuple(i)
        print(
            f"{ti}: diff {diff[ti]} > tol {tol[ti]}"
            f"\tfor values {val_name}={val[ti]}, {val_name}_ref={ref[ti]}"
        )


@pytest.mark.reference
@pytest.mark.layer
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_reference_layer(method, kwargs):
    analyzer_name = method.__name__
    data_path = os.path.join(
        os.path.abspath(os.curdir),
        "tests",
        "references",
        "layer",
        analyzer_name + ".hdf5",
    )

    with h5py.File(data_path, "r") as f:
        assert f.attrs["analyzer_name"] == analyzer_name  # sanity check: correct file
        x = f["input"][:]

        for layer_name, layer in LAYERS_2D.items():
            f_layer = f[layer_name]
            assert f_layer.attrs["layer_name"] == layer_name
            weights = [w[:] for w in f_layer["weights"].values()]

            # Get model
            inputs = keras.layers.Input(shape=input_shape)
            activations = layer(inputs)
            outputs = keras.layers.Flatten()(activations)
            model = keras.Model(inputs=inputs, outputs=outputs, name=layer_name)

            model.set_weights(weights)

            # Model output should match
            y = model.predict(x)
            y_ref = f_layer["output"][:]
            outputs_match = np.allclose(y, y_ref, rtol=rtol, atol=atol)
            if not outputs_match:
                debug_failed_all_close(y, y_ref, "y", layer_name, analyzer_name)
            assert outputs_match

            # Analyze model
            analyzer = method(model, **kwargs)
            a = analyzer.analyze(x)

            # Test attribution
            a_ref = f_layer["attribution"][:]
            attributions_match = np.allclose(a, a_ref, rtol=rtol, atol=atol)
            if not attributions_match:
                debug_failed_all_close(a, a_ref, "a", layer_name, analyzer_name)
            assert attributions_match
