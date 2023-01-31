from __future__ import annotations

import numpy as np
import pytest
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.tools.perturbate
import innvestigate.utils.sequence as isequence


@pytest.mark.perturbate
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PerturbationAnalysis():
    # Some test data
    if kbackend.image_data_format() == "channels_first":
        input_shape = (2, 1, 4, 4)
    else:
        input_shape = (2, 4, 4, 1)
    x = np.arange(2 * 4 * 4).reshape(input_shape)
    generator = isequence.BatchSequence(
        [x, np.zeros(x.shape[0])], batch_size=x.shape[0]
    )

    # Simple model
    model = kmodels.Sequential(
        [
            klayers.Flatten(input_shape=x.shape[1:]),
            klayers.Dense(1, use_bias=False),
        ]
    )

    weights = np.arange(4 * 4 * 1).reshape((4 * 4, 1))
    model.layers[-1].set_weights([weights])
    model.compile(loss="mean_squared_error", optimizer="sgd")

    expected_output = np.array([[1240.0], [3160.0]])
    assert np.all(np.isclose(model.predict(x), expected_output))

    # Analyzer
    analyzer = innvestigate.create_analyzer("gradient", model, postprocess="abs")

    # Run perturbation analysis
    perturbation = innvestigate.tools.perturbate.Perturbation(
        "zeros", region_shape=(2, 2), in_place=False
    )

    perturbation_analysis = innvestigate.tools.perturbate.PerturbationAnalysis(
        analyzer,
        model,
        generator,
        perturbation,
        recompute_analysis=False,
        steps=3,
        regions_per_step=1,
        verbose=False,
    )

    scores = perturbation_analysis.compute_perturbation_analysis()

    expected_scores = np.array([5761600.0, 1654564.0, 182672.0, 21284.0])
    assert np.all(np.isclose(scores, expected_scores))


@pytest.mark.perturbate
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__Perturbation():
    if kbackend.image_data_format() == "channels_first":
        input_shape = (1, 1, 4, 4)
    else:
        input_shape = (1, 4, 4, 1)
    x = np.arange(1 * 4 * 4).reshape(input_shape)

    perturbation = innvestigate.tools.perturbate.Perturbation(
        "zeros", region_shape=(2, 2), in_place=False
    )

    analysis = np.zeros((4, 4))
    analysis[:2, 2:] = 1
    analysis[2:, :2] = 2
    analysis[2:, 2:] = 3
    analysis = analysis.reshape(input_shape)

    if kbackend.image_data_format() == "channels_last":
        x = np.moveaxis(x, 3, 1)
        analysis = np.moveaxis(analysis, 3, 1)

    analysis = perturbation.reduce_function(analysis, axis=1, keepdims=True)

    aggregated_regions = perturbation._aggregate_regions(analysis)
    assert np.all(
        np.isclose(aggregated_regions[0, 0, :, :], np.array([[0, 1], [2, 3]]))
    )

    ranks = perturbation._compute_region_ordering(aggregated_regions)
    assert np.all(np.isclose(ranks[0, 0, :, :], np.array([[3, 2], [1, 0]])))

    perturbation_mask_regions = perturbation._compute_perturbation_mask(ranks, 1)
    assert np.all(perturbation_mask_regions == np.array([[0, 0], [0, 1]]))

    perturbation_mask_regions = perturbation._compute_perturbation_mask(ranks, 4)
    assert np.all(perturbation_mask_regions == np.array([[1, 1], [1, 1]]))

    perturbation_mask_regions = perturbation._compute_perturbation_mask(ranks, 0)
    assert np.all(perturbation_mask_regions == np.array([[0, 0], [0, 0]]))
