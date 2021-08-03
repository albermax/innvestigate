from __future__ import annotations

from typing import Dict, Type

from innvestigate.analyzer.base import AnalyzerBase, NotAnalyzeableModelException
from innvestigate.analyzer.deeptaylor import BoundedDeepTaylor, DeepTaylor
from innvestigate.analyzer.gradient_based import (
    BaselineGradient,
    Deconvnet,
    Gradient,
    GuidedBackprop,
    InputTimesGradient,
    IntegratedGradients,
    SmoothGrad,
)
from innvestigate.analyzer.misc import Input, Random
from innvestigate.analyzer.pattern_based import PatternAttribution, PatternNet
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRP,
    LRPZ,
    BaselineLRPZ,
    LRPAlpha1Beta0,
    LRPAlpha1Beta0IgnoreBias,
    LRPAlpha2Beta1,
    LRPAlpha2Beta1IgnoreBias,
    LRPAlphaBeta,
    LRPEpsilon,
    LRPEpsilonIgnoreBias,
    LRPFlat,
    LRPSequentialPresetA,
    LRPSequentialPresetAFlat,
    LRPSequentialPresetB,
    LRPSequentialPresetBFlat,
    LRPSequentialPresetBFlatUntilIdx,
    LRPWSquare,
    LRPZIgnoreBias,
    LRPZPlus,
    LRPZPlusFast,
)
from innvestigate.analyzer.wrapper import (
    AugmentReduceBase,
    GaussianSmoother,
    PathIntegrator,
    WrapperBase,
)
from innvestigate.utils.types import Model

# Disable pyflaks warnings:
assert NotAnalyzeableModelException
assert BaselineLRPZ
assert WrapperBase
assert AugmentReduceBase
assert GaussianSmoother
assert PathIntegrator


analyzers: Dict[str, Type[AnalyzerBase]] = {
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
    "lrp.sequential_preset_b_flat_until_idx": LRPSequentialPresetBFlatUntilIdx,
    # Deep Taylor
    "deep_taylor": DeepTaylor,
    "deep_taylor.bounded": BoundedDeepTaylor,
    # Pattern based
    "pattern.net": PatternNet,
    "pattern.attribution": PatternAttribution,
}


def create_analyzer(name: str, model: Model, **kwargs) -> AnalyzerBase:
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
            " All possible names are: %s" % (name, list(analyzers.keys()))
        )
    return analyzer_class(model, **kwargs)
