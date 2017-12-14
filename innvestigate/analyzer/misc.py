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


import numpy as np

from .base import BaseAnalyzer


__all__ = ["RandomAnalyzer", "InputAnalyzer"]


class InputAnalyzer(BaseAnalyzer):

    properties = {
        "name": "Input",
        "show_as": "rgb",
    }

    def analyze(self, X):
        return X


class RandomAnalyzer(BaseAnalyzer):

    properties = {
        "name": "Random",
        "show_as": "rgb",
    }

    def analyze(self, X):
        return np.random.randn(*X.shape)
