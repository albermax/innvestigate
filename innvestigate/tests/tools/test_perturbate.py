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


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.precommit
def test_precommit__PerturbationAnalysis():
    # Some test data
    if keras.backend.image_data_format() == "channels_first":
        input_shape = (10, 1, 28, 18)
    else:
        input_shape = (10, 28, 18, 1)
    x = np.random.rand(*input_shape)

    # Simple model
    model = keras.models.Sequential([
            keras.layers.Dense(10, input_shape=x.shape[1:]),
    ])

    # Run perturbation analysis
    pass
