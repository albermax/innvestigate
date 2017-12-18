
from .base import *

from .gradient_based import *
from .misc import *
#from .pattern_based import *
#from .relevance_based import *


def create_analyzer(name, moderl, **kwargs):
    return {
        # Utility.
        "input": InputAnalyzer,
        "random": RandomAnalyzer,

        # # Gradient based
        # "gradient": GradientAnalyzer,
        # "deconvnet": DeConvNetAnalyzer,
        # "guided": GuidedBackpropAnalyzer,
        "gradient.baseline": BaselineGradientAnalyzer,

        # # Relevance based
        # "lrp.z": LRPZAnalyzer,
        # "lrp.eps": LRPEpsAnalyzer,

        # # Pattern based
        # "patternnet": PatternNetAnalyzer,
        # "patternnet.guided": GuidedPatternNetAnalyzer,
        # "patternlrp": PatternLRPAnalyzer,
    }[name](model **kwargs)
