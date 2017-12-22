# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
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


__all__ = [
    "easy_apply",
]


###############################################################################
###############################################################################
###############################################################################


def easy_apply(layer, inputs):
    """
    Apply a layer to input[s].
    """
    try:
        ret = layer(inputs)
    except (TypeError, AttributeError):
        # layer expects a single tensor.
        if len(inputs) != 1:
            raise ValueError("Layer expects only a single input!")
        ret = [layer(inputs[0])]
    return ret
