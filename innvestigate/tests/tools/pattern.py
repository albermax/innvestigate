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

from innvestigate.tools import PatternComputer


###############################################################################
###############################################################################
###############################################################################


class TestPatterComputer_dummy_parallel(dryrun.PatternComputerTestCase):

    def _method(self, model):
        return PatternComputer(model, pattern_type="dummy",
                               compute_layers_in_parallel=True)


class TestPatterComputer_dummy_sequential(dryrun.PatternComputerTestCase):

    def _method(self, model):
        return PatternComputer(model, pattern_type="dummy",
                               compute_layers_in_parallel=False)
