import pytest


# Import all test cases
from .trivia import dot
from .trivia import skip_connection

# MLPs
from .mlp import mlp2
from .mlp import mlp3

# CNNs
from .cnn import cnn_1dim_c1_d1
from .cnn import cnn_1dim_c2_d1
from .cnn import cnn_2dim_c1_d1
from .cnn import cnn_2dim_c2_d1
from .cnn import cnn_3dim_c1_d1
from .cnn import cnn_3dim_c2_d1
# locally connected CNNs
from .cnn import lc_cnn_1dim_c1_d1
from .cnn import lc_cnn_1dim_c2_d1
from .cnn import lc_cnn_2dim_c1_d1
from .cnn import lc_cnn_2dim_c2_d1

# Special layers
from .special import batchnorm
from .special import dropout


# Convenience lists of test cases.
FAST = [
    "dot",
    "skip_connection",

    "mlp2",

    "cnn_2dim_c1_d1",
    "cnn_2dim_c2_d1",

    "batchnorm",
    "dropout",
]

PRECOMMIT = [
    "mlp3",

    "cnn_1dim_c1_d1",
    "cnn_1dim_c2_d1",
    "cnn_3dim_c1_d1",
    "cnn_3dim_c2_d1",

    "lc_cnn_1dim_c1_d1",
    "lc_cnn_1dim_c2_d1",
    "lc_cnn_2dim_c1_d1",
    "lc_cnn_2dim_c2_d1",
]


def mark_as_xfail(case_ids, xfails):
    """Mark cases as expected failures.

    :param case_ids: Parameter list for pytest.mar.parametrize.
    :param xfails: Parameters in case_ids to mark as expected failures.
    :return: case_ids with added marks.
    """
    ret = []
    for case in case_ids:
        if case in xfails:
            if not isinstance(case, tuple):
                case = (case,)
            # Mark as expected failure
            case = pytest.param(*case, marks=pytest.mark.xfail)
        ret.append(case)
    return ret
