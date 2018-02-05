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


import keras.utils
import math


__all__ = [
    "listify",
    "BatchSequence",
]


###############################################################################
###############################################################################
###############################################################################


def listify(l):
    if not isinstance(l, list):
        return [l, ]
    else:
        return l


###############################################################################
###############################################################################
###############################################################################


class BatchSequence(keras.utils.Sequence):

    def __init__(self, X, batch_size=32):
        self.X = X
        self.batch_size = batch_size
        super(BatchSequence, self).__init__()

    def __len__(self):
        return int(math.ceil(float(len(self.X)) / self.batch_size))

    def __getitem__(self, idx):
        return self.X[idx*self.batch_size:(idx+1)*self.batch_size]
