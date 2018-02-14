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


# todo:fix relative imports:
#from ...utils.tests import dryrun

#from ...explainer import InputExplainer
#from ...explainer import RandomExplainer

from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import PatternNet
from innvestigate.analyzer import PatternAttribution


###############################################################################
###############################################################################
###############################################################################


class TestPatternNet(dryrun.AnalyzerTestCase):

    def _method(self, model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)


class TestPatternAttribution(dryrun.AnalyzerTestCase):

    def _method(self, model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)


###############################################################################
###############################################################################
###############################################################################


class TestSerializePatternNet(dryrun.SerializeAnalyzerTestCase):

    def _method(self, model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)
