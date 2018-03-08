# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import \
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


from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import BaselineGradient
from innvestigate.analyzer import Gradient


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


class TestBaseReverseNetwork_reverse_debug(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return Gradient(model, reverse_verbose=True)


class TestBaseReverseNetwork_reverse_check_minmax(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return Gradient(model, reverse_verbose=True,
                        reverse_check_min_max_values=True)


class TestBaseReverseNetwork_reverse_check_finite(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return Gradient(model, reverse_verbose=True, reverse_check_finite=True)


###############################################################################
###############################################################################
###############################################################################


class TestSerializeAnalyzerBase(dryrun.SerializeAnalyzerTestCase):

    def _method(self, model):
        return BaselineGradient(model)


class TestSerializeReverseAnalyzerkBase(dryrun.SerializeAnalyzerTestCase):

    def _method(self, model):
        return Gradient(model)
