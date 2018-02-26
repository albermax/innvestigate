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

from innvestigate.analyzer import Input
from innvestigate.analyzer import Random


###############################################################################
###############################################################################
###############################################################################


class TestInput(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return Input(model)


class TestRandom(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return Random(model)
