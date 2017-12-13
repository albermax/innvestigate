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

# Todo: fix:
#from ...utils.tests import dryrun
from innvestigate.utils.tests import dryrun

class TestDryRunExplainerTestCase(dryrun.ExplainerTestCase):
    """
    Sanity test for the TestCase.
    """

    def _method(self, output_layer):

        class TestExplainer(object):
            def explain(self, X):
                return X

        return TestExplainer()
