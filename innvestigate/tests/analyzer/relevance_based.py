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

from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import *


###############################################################################
###############################################################################
###############################################################################


class TestBaselineLRPZ(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return BaselineLRPZ(model)


###############################################################################
###############################################################################
###############################################################################


class TestLRPZ(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPZ(model)


class TestLRPZ__equal_BaselineLRPZ(dryrun.EqualAnalyzerTestCase):

    def _method1(self, model):
        return BaselineLRPZ(model)

    def _method2(self, model):
        return LRPZ(model)


class TestLRPZ__with_input_layer_rule(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPZ(model, input_layer_rule="Flat")


class TestLRPZ__with_boxed_input_layer_rule(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPZ(model, input_layer_rule=(-10, 10))


class TestLRPZPlus(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPZPlus(model)


class TestLRPEpsilon(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPEpsilon(model)


class TestLRPWSquare(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPWSquare(model)


class TestLRPFlat(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPFlat(model)


class TestLRPAlphaBeta(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPAlphaBeta(model)


class TestLRPAlpha1Beta1(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPAlpha1Beta1(model)


class TestLRPAlpha2Beta1(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPAlpha2Beta1(model)


class TestLRPAlpha1Beta0(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return LRPAlpha1Beta0(model)
