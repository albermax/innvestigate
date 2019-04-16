# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


import os


try:
    # Try to import TensorFlow,
    import tensorflow as tf
    # if possible, import our backend.
    from . import tensorflow as tf_utils
except ImportError:
    tf = None


try:
    # Try to import PyTorch,
    import torch as torch
    # if possible, import our backend.
    from . import torch as torch_utils
except ImportError:
    torch = None


if tf and torch:
    # Both backends are installed,
    # let's choose one.
    __backend = os.environ.get("INNVESTIGATE_BACKEND", "tensorflow")
    if __backend not in ("tensorflow", "torch"):
        raise ValueError("Env-variable INNVESTIGATE_BACKEND must be set to "
                         "either 'tensorflow' or 'torch'.")

elif tf:
    __backend = "tensorflow"
else:
    __backend = "torch"


if __backend == "tensorflow":
    torch = None
    torch_utils = None
else:
    tf = None
    tf_utils = None


def get_name():
    return __backend
