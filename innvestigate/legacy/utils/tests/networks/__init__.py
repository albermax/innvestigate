# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import keras.backend as K
import fnmatch

from . import trivia
from . import mnist
from . import cifar10
from . import imagenet


###############################################################################
###############################################################################
###############################################################################


def iterator(network_filter="*", clear_sessions=False):
    """
    Iterator over various networks.
    """

    def fetch_networks(module_name, module):
        ret = [
            ("%s.%s" % (module_name, name),
             (module, name))
            for name in module.__all__
            if any((fnmatch.fnmatch(name, one_filter) or
                    fnmatch.fnmatch("%s.%s" % (module_name, name), one_filter))
                   for one_filter in network_filter.split(":"))
        ]

        return [x for x in sorted(ret)]

    networks = (
        fetch_networks("trivia", trivia) +
        fetch_networks("mnist", mnist) +
        fetch_networks("cifar10", cifar10) +
        fetch_networks("imagenet", imagenet)
    )

    for module_name, (module, name) in networks:
        if clear_sessions:
            K.clear_session()

        network = getattr(module, name)()
        network["name"] = module_name
        yield network
