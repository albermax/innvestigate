# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################

from .base import NotAnalyzeableModelException
from .new_base import ReverseAnalyzerBase
from .deeplift import DeepLIFTWrapper
from .gradient_based import BaselineGradient
from .gradient_based import Gradient
from .gradient_based import InputTimesGradient
from .gradient_based import GuidedBackprop
from .gradient_based import Deconvnet
from .gradient_based import IntegratedGradients
from .gradient_based import SmoothGrad
from .misc import Input
from .misc import Random
from .pattern_based import PatternNet
from .pattern_based import PatternAttribution
from .relevance_based.relevance_analyzer import BaselineLRPZ
from .relevance_based.relevance_analyzer import LRP
from .relevance_based.relevance_analyzer import LRPZ
from .relevance_based.relevance_analyzer import LRPZIgnoreBias
from .relevance_based.relevance_analyzer import LRPZPlus
from .relevance_based.relevance_analyzer import LRPZPlusFast
from .relevance_based.relevance_analyzer import LRPEpsilon
from .relevance_based.relevance_analyzer import LRPEpsilonIgnoreBias
from .relevance_based.relevance_analyzer import LRPWSquare
from .relevance_based.relevance_analyzer import LRPFlat
from .relevance_based.relevance_analyzer import LRPAlphaBeta
from .relevance_based.relevance_analyzer import LRPAlpha2Beta1
from .relevance_based.relevance_analyzer import LRPAlpha2Beta1IgnoreBias
from .relevance_based.relevance_analyzer import LRPAlpha1Beta0
from .relevance_based.relevance_analyzer import LRPAlpha1Beta0IgnoreBias
from .relevance_based.relevance_analyzer import LRPSequentialPresetA
from .relevance_based.relevance_analyzer import LRPSequentialPresetB
from .relevance_based.relevance_analyzer import LRPSequentialPresetAFlat
from .relevance_based.relevance_analyzer import LRPSequentialPresetBFlat
from .relevance_based.relevance_analyzer_new import LRP as LRP_new
from .relevance_based.relevance_analyzer_new import LRPZ as LRPZ_new
from .relevance_based.relevance_analyzer_new import LRPZIgnoreBias as LRPZIgnoreBias_new
from .relevance_based.relevance_analyzer_new import LRPEpsilon as LRPEpsilon_new
from .relevance_based.relevance_analyzer_new import LRPEpsilonIgnoreBias as LRPEpsilonIgnoreBias_new
from .relevance_based.relevance_analyzer_new import LRPWSquare as LRPWSquare_new
from .relevance_based.relevance_analyzer_new import LRPFlat as LRPFlat_new
from .relevance_based.relevance_analyzer_new import LRPAlphaBeta as LRPAlphaBeta_new
from .relevance_based.relevance_analyzer_new import LRPAlpha2Beta1 as LRPAlpha2Beta1_new
from .relevance_based.relevance_analyzer_new import LRPAlpha2Beta1IgnoreBias as LRPAlpha2Beta1IgnoreBias_new
from .relevance_based.relevance_analyzer_new import LRPAlpha1Beta0 as LRPAlpha1Beta0_new
from .relevance_based.relevance_analyzer_new import LRPAlpha1Beta0IgnoreBias as LRPAlpha1Beta0IgnoreBias_new
from .relevance_based.relevance_analyzer_new import LRPSequentialPresetA as LRPSequentialPresetA_new
from .relevance_based.relevance_analyzer_new import LRPSequentialPresetB as LRPSequentialPresetB_new
from .relevance_based.relevance_analyzer_new import LRPSequentialPresetAFlat as LRPSequentialPresetAFlat_new
from .relevance_based.relevance_analyzer_new import LRPSequentialPresetBFlat as LRPSequentialPresetBFlat_new
from .relevance_based.relevance_analyzer_new import LRPSequentialCompositeA as LRPSequentialCompositeA_new
from .relevance_based.relevance_analyzer_new import LRPSequentialCompositeB as LRPSequentialCompositeB_new
from .relevance_based.relevance_analyzer_new import LRPSequentialCompositeAFlat as LRPSequentialCompositeAFlat_new
from .relevance_based.relevance_analyzer_new import LRPSequentialCompositeBFlat as LRPSequentialCompositeBFlat_new
from .deeptaylor import DeepTaylor
from .deeptaylor import BoundedDeepTaylor
from .wrapper import WrapperBase
from .wrapper import AugmentReduceBase
from .wrapper import GaussianSmoother
from .wrapper import PathIntegrator


# Disable pyflaks warnings:
assert NotAnalyzeableModelException
assert BaselineLRPZ
assert WrapperBase
assert AugmentReduceBase
assert GaussianSmoother
assert PathIntegrator


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
    "deep_lift.wrapper": DeepLIFTWrapper,

    # Pattern based
    "pattern.net": PatternNet,
    "pattern.attribution": PatternAttribution,
}


def create_analyzer(name, model, **kwargs):
    """Instantiates the analyzer with the name 'name'

    This convenience function takes an analyzer name
    creates the respective analyzer.

    Alternatively analyzers can be created directly by
    instantiating the respective classes.

    :param name: Name of the analyzer.
    :param model: The model to analyze, passed to the analyzer's __init__.
    :param kwargs: Additional parameters for the analyzer's .
    :return: An instance of the chosen analyzer.
    :raise KeyError: If there is no analyzer with the passed name.
    """
    try:
        analyzer_class = analyzers[name]
    except KeyError:
        raise KeyError(
            "No analyzer with the name '%s' could be found."
            " All possible names are: %s" % (name, list(analyzers.keys())))
    return analyzer_class(model, **kwargs)
