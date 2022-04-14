"""
Test all LRP Analyzers on VGG16.
"""
# NOTE:
# For VGG16 reference tests to work, clone the repo
# https://github.com/adrhill/test-data-innvestigate
# and run the file `test-data-innvestigate/src/generate/vgg16.py`.
#
# Using poetry, you can alternatively run
# ```bash
# poetry run test-data-innvestigate
# ```
#
# This will generate ~50 MB of reference files in `test-data-innvestigate/data/vgg16`
# that then need to be copied into `innvestigate/tests/references/vgg16`.
#
# By default, these tests are excluded from pytest runs and will only be run
# when including the marker `local`, e.g. `poetry run pytest -m local`.

import os

import h5py
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.applications import VGG16

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
    LRP,
    LRPZ,
    LRPAlpha1Beta0,
    LRPAlpha1Beta0IgnoreBias,
    LRPAlpha2Beta1,
    LRPAlpha2Beta1IgnoreBias,
    LRPEpsilon,
    LRPFlat,
    LRPSequentialPresetA,
    LRPSequentialPresetAFlat,
    LRPSequentialPresetB,
    LRPSequentialPresetBFlat,
    LRPWSquare,
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
    "LRPZPlus": (LRPZPlus, {}),
    "LRPZPlusFast": (LRPZPlusFast, {}),
    "LRPAlpha1Beta0": (LRPAlpha1Beta0, {}),
    "LRPAlpha1Beta0IgnoreBias": (LRPAlpha1Beta0IgnoreBias, {}),
    "LRPAlpha2Beta1": (LRPAlpha2Beta1, {}),
    "LRPAlpha2Beta1IgnoreBias": (LRPAlpha2Beta1IgnoreBias, {}),
    "LRPEpsilon": (LRPEpsilon, {}),
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

rtol = 1e-2
atol = 1e-5

# Loosen tolerances for SmoothGrad because of random Gaussian noise
atol_smoothgrad = 0.15
rtol_smoothgrad = 0.25

# Sizes used for data generation
INPUT_SHAPE = (10, 10, 3)
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)


def debug_failed_all_close(val, ref, val_name, analyzer_name, rtol=rtol, atol=atol):
    diff = np.absolute(val - ref)
    # Function evaluated by np.allclose, see "Notes":
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    tol = atol + rtol * np.absolute(ref)
    idx = np.argwhere(diff > tol)

    print(
        f"{len(idx)}/{np.prod(val.shape)} "
        f'failed on reference "{val_name}" using {analyzer_name}'
        f"(atol={atol}, rtol={rtol})"
    )


@pytest.mark.local
@pytest.mark.parametrize(
    "analyzer_name, val", methods.items(), ids=list(methods.keys())
)
def test_reference_layer(val, analyzer_name):
    method, kwargs = val
    data_path = os.path.join(
        os.path.abspath(os.curdir),
        "tests",
        "references",
        "vgg16",
        analyzer_name + ".hdf5",
    )

    tf.keras.backend.clear_session()

    # Load model from keras.applications
    model = VGG16(classifier_activation=None)

    with h5py.File(data_path, "r") as f:
        assert f.attrs["analyzer_name"] == analyzer_name  # sanity check: correct file
        # Model outputs should match
        x = f["input"][:]
        y = model.predict(x)

        y_ref = f["output"][:]
        assert np.allclose(y, y_ref, rtol=rtol, atol=atol)

        # Analyze model
        analyzer = method(model, **kwargs)
        if isinstance(analyzer, LRP):
            analyzer._reverse_keep_tensors = True

        a = analyzer.analyze(x)
        assert np.shape(a) == np.shape(x)

        # Set tolerances for reference tests
        if analyzer_name == "SmoothGrad":
            _atol = atol_smoothgrad
            _rtol = rtol_smoothgrad
        else:
            _atol = atol
            _rtol = rtol

        # Test reverse tensors
        if isinstance(analyzer, LRP):
            relevances = analyzer._reversed_tensors
            # unzip reverse tensors to strip indices
            indices, relevances = zip(*relevances)

            for i, r in zip(reversed(indices), reversed(relevances)):
                idx = str(i[0])
                r_ref = f["layerwise_relevances"][idx][:]
                rels_match = np.allclose(r, r_ref, rtol=_rtol, atol=_atol)
                if not rels_match:
                    debug_failed_all_close(
                        r, r_ref, f"r_{idx}", analyzer_name, rtol=_rtol, atol=_atol
                    )
                else:
                    print(f"r_{idx} passed")
                assert rels_match

        # Test attribution
        a_ref = f["attribution"][:]
        attributions_match = np.allclose(a, a_ref, rtol=_rtol, atol=_atol)
        if not attributions_match:
            debug_failed_all_close(a, a_ref, "a", analyzer_name, rtol=_rtol, atol=_atol)
        assert attributions_match
