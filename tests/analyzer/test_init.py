"""Test function 'innvestigate.create_analyzer'"""
from __future__ import annotations

import logging

import pytest
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

from innvestigate import create_analyzer
from innvestigate.analyzer import analyzers


@pytest.mark.init
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__create_analyzers():
    """
    Test 'innvestigate.create_analyzer':
    Instantiate analyzers by name using a placeholder Keras model.
    """

    fake_model = kmodels.Sequential([klayers.Dense(10, input_shape=(10,))])
    for name in analyzers:
        try:
            create_analyzer(name, fake_model)
        except KeyError:
            print("Key not found when creating analyzer from name.")
        except Exception:
            logging.error("Error when creating analyzer from name.", exc_info=True)


@pytest.mark.init
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__create_analyzers_wrong_name():
    """
    Test 'innvestigate.create_analyzer':
    'KeyError' should be thrown when passing wrong keys.
    """
    fake_model = kmodels.Sequential([klayers.Dense(10, input_shape=(10,))])
    with pytest.raises(KeyError):
        create_analyzer("wrong name", fake_model)
