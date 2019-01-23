# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import \
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import pytest

import keras.layers
import keras.models

from innvestigate import create_analyzer
from innvestigate.analyzer import analyzers


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__create_analyzers():

    fake_model = keras.models.Sequential([
        keras.layers.Dense(10, input_shape=(10,))
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

    fake_model = keras.models.Sequential([
        keras.layers.Dense(10, input_shape=(10,))
    ])
    with pytest.raises(KeyError) as e_info:
        create_analyzer("wrong name", fake_model)
