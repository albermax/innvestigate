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

from innvestigate.analyzer import BaselineGradient
from innvestigate.analyzer import Gradient

from innvestigate.analyzer import Deconvnet
from innvestigate.analyzer import GuidedBackprop


###############################################################################
###############################################################################
###############################################################################


class TestBaselineGradient(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return BaselineGradient(model)


class TestGradient(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return Gradient(model)


###############################################################################
###############################################################################
###############################################################################


class TestBasicGraphReversal(dryrun.EqualAnalyzerTestCase):

    def _method1(self, model):
        return BaselineGradient(model)

    def _method2(self, model):
        return Gradient(model)


###############################################################################
###############################################################################
###############################################################################


class TestDeconvnet(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return Deconvnet(model)


class TestGuidedBackprop(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return GuidedBackprop(model)
