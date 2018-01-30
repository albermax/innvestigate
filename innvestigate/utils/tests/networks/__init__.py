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


import fnmatch
import os

from . import trivia
from . import mnist
from . import cifar10
from . import imagenet


###############################################################################
###############################################################################
###############################################################################


def iterator():
    """
    Iterator over various networks.
    """

    # TODO: change environment variable name.
    # TODO: make this more transparent!
    # Default test only for one network. To test all put "*"
    #name_filter = "mnist.log_reg"
    name_filter = "*"
    if "NNPATTERNS_TEST_FILTER" in os.environ:
        name_filter = os.environ["NNPATTERNS_TEST_FILTER"]

    def fetch_networks(module_name, module):
        ret = [
            ("%s.%s" % (module_name, name),
             getattr(module, name)())
            for name in module.__all__
            if (fnmatch.fnmatch(name, name_filter) or
                fnmatch.fnmatch("%s.%s" % (module_name, name), name_filter))
        ]

        for name, network in ret:
            network["name"] = name

        return [x[1] for x in sorted(ret)]

    networks = (
        fetch_networks("trivia", trivia) +
        fetch_networks("mnist", mnist) +
        fetch_networks("cifar10", cifar10) +
        fetch_networks("imagenet", imagenet)
    )

    for network in networks:
        yield network
