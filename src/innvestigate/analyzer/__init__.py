from __future__ import annotations

from innvestigate.analyzer.base import AnalyzerBase
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
    LRPFlat,
    LRPSequentialPresetA,
    LRPSequentialPresetAFlat,
    LRPSequentialPresetB,
    LRPSequentialPresetBFlat,
    LRPSequentialPresetBFlatUntilIdx,
    LRPWSquare,
    LRPZPlus,
    LRPZPlusFast,
)
from innvestigate.analyzer.wrapper import AugmentReduceBase  # noqa
from innvestigate.analyzer.wrapper import GaussianSmoother  # noqa
from innvestigate.analyzer.wrapper import PathIntegrator  # noqa
from innvestigate.analyzer.wrapper import WrapperBase  # noqa
from innvestigate.backend.types import Model

# Silence ruff
assert BaselineLRPZ

analyzers: dict[str, type[AnalyzerBase]] = {
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
    "lrp.epsilon": LRPEpsilon,
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
            f"No analyzer with the name '{name}' could be found."
            f" All possible names are: {list(analyzers.keys())}"
        )
    return analyzer_class(model, **kwargs)
