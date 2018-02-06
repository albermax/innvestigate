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
        "gradient.baseline": BaselineGradient,
        "deconvnet": Deconvnet,
        "guided_backprop": GuidedBackprop,

        # Relevance based
        "lrp": LRP,
        "lrp.z_baseline": BaselineLRPZ,
        "lrp.z": LRPZ,
        "lrp.epsilon": LRPEpsilon,
        "lrp.w_square": LRPWSquare,
        "lrp.flat": LRPFlat,
        "lrp.alpha_beta": LRPAlphaBeta,
        "lrp.alpha_1_beta_1": LRPAlpha1Beta1,
        "lrp.alpha_2_beta_1": LRPAlpha2Beta1,
        "lrp.alpha_1_beta_0": LRPAlpha1Beta0,

        # Pattern based
        "pattern.net": PatternNet,
        "pattern.attribution": PatternAttribution,
    }[name](model, **kwargs)
