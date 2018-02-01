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


from .base import *

from .gradient_based import *
from .misc import *
from .pattern_based import *
from .relevance_based import *


###############################################################################
###############################################################################
###############################################################################


def create_analyzer(name, model, **kwargs):
    return {
        # Utility.
        "input": Input,
        "random": Random,

        # Gradient based
        "gradient": Gradient,
        "deconvnet": Deconvnet,
        "guided": GuidedBackprop,
        "gradient.baseline": BaselineGradient,

        # Relevance based
        "lrp.z_baseline": BaselineLRPZ,
        #"lrp.eps": LRPEps,

        # Pattern based
        "pattern.net": PatternNet,
        "pattern.attribution": PatternAttribution,
    }[name](model, **kwargs)
