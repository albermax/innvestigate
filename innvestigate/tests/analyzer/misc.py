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

from innvestigate.analyzer import InputAnalyzer
from innvestigate.analyzer import RandomAnalyzer


###############################################################################
###############################################################################
###############################################################################


class TestInputAnalyzer(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return InputAnalyzer(model)


class TestRandomAnalyzer(dryrun.AnalyzerTestCase):

    def _method(self, model):
        return RandomAnalyzer(model)
