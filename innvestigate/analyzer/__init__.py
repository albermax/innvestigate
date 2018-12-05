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
from .wrapper import *

from .deeplift import *
from .gradient_based import *
from .misc import *
from .pattern_based import *
from .relevance_based.relevance_analyzer import *
from .deeptaylor import *



###############################################################################
###############################################################################
###############################################################################


analyzers = {
    # Utility.
    "input": Input,
    "random": Random,

    # Gradient based
    "gradient": Gradient,
    "gradient.baseline": BaselineGradient,
    "input_t_gradient": InputTimesGradient,
    "deconvnet": Deconvnet,
    "guided_backprop": GuidedBackprop,
    "integrated_gradients": IntegratedGradients,
    "smoothgrad": SmoothGrad,

    # Relevance based
    "lrp": LRP,
    "lrp.z": LRPZ,
    "lrp.z_IB": LRPZIgnoreBias,

    "lrp.epsilon": LRPEpsilon,
    "lrp.epsilon_IB": LRPEpsilonIgnoreBias,

    "lrp.w_square": LRPWSquare,
    "lrp.flat": LRPFlat,

    "lrp.alpha_beta": LRPAlphaBeta,

    "lrp.alpha_2_beta_1": LRPAlpha2Beta1,
    "lrp.alpha_2_beta_1_IB": LRPAlpha2Beta1IgnoreBias,
    "lrp.alpha_1_beta_0": LRPAlpha1Beta0,
    "lrp.alpha_1_beta_0_IB": LRPAlpha1Beta0IgnoreBias,
    "lrp.z_plus": LRPZPlus,
    "lrp.z_plus_fast": LRPZPlusFast,

    "lrp.sequential_preset_a": LRPSequentialPresetA,
    "lrp.sequential_preset_b": LRPSequentialPresetB,
    "lrp.sequential_preset_a_flat": LRPSequentialPresetAFlat,
    "lrp.sequential_preset_b_flat": LRPSequentialPresetBFlat,

    # Deep Taylor
    "deep_taylor": DeepTaylor,
    "deep_taylor.bounded": BoundedDeepTaylor,

    # DeepLIFT
    "deep_lift": DeepLIFT,
    "deep_lift.wrapper": DeepLIFTWrapper,

    # Pattern based
    "pattern.net": PatternNet,
    "pattern.attribution": PatternAttribution,
}


def create_analyzer(name, model, **kwargs):
    """ Convenience interface to create analyzers.

    This function is a convenient interface to create analyzer.
    It allows to address analyzers via names instead of classes.

    :param name: Name of the analyzer.
    :param model: The model to analyze.
    :param kwargs: Parameters for the analyzer's init function.
    :return: An instance of the chosen analyzer.
    """
    return analyzers[name](model, **kwargs)
