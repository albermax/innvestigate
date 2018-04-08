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


from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import WrapperBase
from innvestigate.analyzer import AugmentReduceBase
from innvestigate.analyzer import GaussianSmoother
from innvestigate.analyzer import PathIntegrator

from innvestigate.analyzer import Input
from innvestigate.analyzer import Gradient


###############################################################################
###############################################################################
###############################################################################


class TestWrapperBase(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return WrapperBase(Gradient(model))


class TestSerializeWrapperBase(dryrun.SerializeAnalyzerTestCase):

    def _method(self, model):
        return WrapperBase(Gradient(model))


###############################################################################
###############################################################################
###############################################################################


class TestAugmentReduceBase__python_based(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return AugmentReduceBase(Input(model))


class TestAugmentReduceBase__keras_based(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return AugmentReduceBase(Gradient(model))


class TestSerializeAugmentReduceBase(dryrun.SerializeAnalyzerTestCase):

    def _method(self, model):
        return AugmentReduceBase(Gradient(model))


###############################################################################
###############################################################################
###############################################################################


class TestGaussianSmoother__python_based(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return GaussianSmoother(Input(model))


class TestGaussianSmoother__keras_based(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return GaussianSmoother(Gradient(model))


class TestSerializeGaussianSmoother(dryrun.SerializeAnalyzerTestCase):

    def _method(self, model):
        return GaussianSmoother(Gradient(model))


###############################################################################
###############################################################################
###############################################################################


class TestPathIntegrator__python_based(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return PathIntegrator(Input(model))


class TestPathIntegrator__keras_based(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return PathIntegrator(Gradient(model))


class TestSerializePathIntegrator(dryrun.SerializeAnalyzerTestCase):

    def _method(self, model):
        return PathIntegrator(Gradient(model))
