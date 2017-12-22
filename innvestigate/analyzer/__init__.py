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
#from .pattern_based import *
#from .relevance_based import *


###############################################################################
###############################################################################
###############################################################################


def create_analyzer(name, moderl, **kwargs):
    return {
        # Utility.
        "input": InputAnalyzer,
        "random": RandomAnalyzer,

        # # Gradient based
        # "gradient": GradientAnalyzer,
        # "deconvnet": DeconvnetAnalyzer,
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
