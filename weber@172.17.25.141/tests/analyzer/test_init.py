# Get Python six functionality:
from __future__ import \
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest

import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models

from innvestigate import create_analyzer
from innvestigate.analyzer import analyzers


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__create_analyzers():

    fake_model = keras_models.Sequential([
        keras_layers.Dense(10, input_shape=(10,))
    ])
    for name in analyzers.keys():
        try:
            create_analyzer(name, fake_model)
        except KeyError:
            # Name should be found!
            raise
        except:
            # Some analyzers require parameters...
            pass


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__create_analyzers_wrong_name():

    fake_model = keras_models.Sequential([
        keras_layers.Dense(10, input_shape=(10,))
    ])
    with pytest.raises(KeyError):
        create_analyzer("wrong name", fake_model)
