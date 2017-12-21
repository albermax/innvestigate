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


from . import base
from .. import layers as ilayers
from .. import utils

import keras.backend as K
import keras.models


__all__ = [
    "BaselineGradientAnalyzer",
    "GradientAnalyzer",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineGradientAnalyzer(base.BaseNetworkAnalyzer):

    properties = {
        "name": "BaselineGradient",
        "show_as": "rgb",
    }


    def _create_analysis(self, model):
        return ilayers.Gradient()(model.inputs+[model.outputs[0],])


class GradientAnalyzer(base.BaseReverseNetworkAnalyzer):

    properties = {
        "name": "Gradient",
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        # we assume there is only one head!
        gradient_head_processed = [False]
        def gradient_reverse(Xs, Ys, reversed_Ys, reverse_state):
            if gradient_head_processed[0] is not True:
                # replace function value with ones as the last element
                # chain rule is a one.
                gradient_head_processed[0] = True
                reversed_Ys = utils.listify(ilayers.OnesLike()(reversed_Ys))
            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

        self.default_reverse = gradient_reverse
        return super(GradientAnalyzer, self).__init__(*args, **kwargs)
