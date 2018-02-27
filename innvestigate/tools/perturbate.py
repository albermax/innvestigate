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

import numpy as np

import keras.backend as K


# TODO
class Perturbation:
    def __init__(self, perturbation_function, steps, recompute_analysis=False):
        raise NotImplementedError  # TODO

    def sort(self):
        pass
        raise NotImplementedError  # TODO

    def perturbate(self):
        raise NotImplementedError  # TODO

    def generator(*args, **kwargs):
        raise NotImplementedError  # TODO

    def compute(self, X, batch_size=32, verbose=0):
        raise NotImplementedError  # TODO
        generator = None
        return self.compute_generator(generator, verbose=verbose)

    def compute_generator(self, generator, **kwargs):
        raise NotImplementedError  # TODO
