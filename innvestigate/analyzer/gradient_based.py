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


__all__ = ["BaselineGradientAnalyzer"]


from . import base
from .. import layers as ilayers

import keras.backend as K


class BaselineGradientAnalyzer(base.BaseNetworkAnalyzer):
    
    properties = {
        "name": "BaselineGradient",
        "show_as": "rgb",
    }


    def _create_analysis(self, model):
        return ilayers.Gradient()(model.inputs+[model.outputs[0],])
        import keras.layers
        ret = keras.layers.Lambda(lambda x: K.gradients(x[1].sum(), x[0]))([model.inputs[0], model.outputs[0]])
        print(type(ret[0]))
        return ret[0]