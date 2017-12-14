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


__all__ = ["BaseAnalyzer"]


class BaseAnalyzer(object):

    properties = {
        "name": "undefined",
        "show_as": "undefined",
    }

    def __init__(self, model):
        self._model = model
        pass

    def explain(self, X):
        raise NotImplementedError("Has to be implemented by the subclass")
