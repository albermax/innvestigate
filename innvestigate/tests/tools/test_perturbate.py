# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import range, zip


###############################################################################
###############################################################################
###############################################################################


import keras.layers
import keras.models
import numpy as np
import pytest

import innvestigate.tools.perturbate
import innvestigate.utils as iutils


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.precommit
def test_precommit__PerturbationAnalysis():
    # Some test data
    if keras.backend.image_data_format() == "channels_first":
        input_shape = (2, 1, 4, 4)
    else:
        input_shape = (2, 4, 4, 1)
    x = np.arange(2 * 4 * 4).reshape(input_shape)
    generator = iutils.BatchSequence([x, np.zeros(x.shape[0])], batch_size=x.shape[0])

    # Simple model
    model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=x.shape[1:]),
            keras.layers.Dense(1, use_bias=False),
    ])

    weights = np.arange(4 * 4 * 1).reshape((4 * 4, 1))
    model.layers[-1].set_weights([weights])
    model.compile(loss='mean_squared_error', optimizer='sgd')

    expected_output = np.array([[1240.], [3160.]])
    assert np.all(np.isclose(model.predict(x), expected_output))

    # Analyzer
    analyzer = innvestigate.create_analyzer("gradient",
                                              model,
                                              postprocess="abs")

    # Run perturbation analysis
    perturbation = innvestigate.tools.perturbate.Perturbation("zeros", region_shape=(2, 2), in_place=False)

    perturbation_analysis = innvestigate.tools.perturbate.PerturbationAnalysis(analyzer, model, generator, perturbation, recompute_analysis=False,
                                                 steps=3, regions_per_step=1, verbose=False)

    scores = perturbation_analysis.compute_perturbation_analysis()

    expected_scores = np.array([5761600.0, 1654564.0, 182672.0, 21284.0])
    assert np.all(np.isclose(scores, expected_scores))
