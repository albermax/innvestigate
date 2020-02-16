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


def _mark_cases(case_ids, to_mark, mark):
    """Mark cases.

    :param case_ids: Parameter list for pytest.mark.parametrize.
    :param xfails: List of parameters in case_ids to mark.
    :param mark: Mark to apply.
    :return: case_ids with added marks.
    """
    ret = []
    for case in case_ids:
        if case in to_mark:
            if not isinstance(case, tuple):
                case = (case,)
            # Mark as expected failure
            case = pytest.param(*case, marks=mark)
        ret.append(case)
    return ret


def mark_as_xfail(case_ids, xfails):
    """Mark cases as expected failures.

    :param case_ids: Parameter list for pytest.mark.parametrize.
    :param xfails: List of parameters in case_ids to mark as expected failures.
    :return: case_ids with added marks.
    """
    return _mark_cases(case_ids, xfails, pytest.mark.xfail)


def mark_as_skip(case_ids, skips):
    """Mark cases to skip.

    :param case_ids: Parameter list for pytest.mark.parametrize.
    :param xfails: List of parameters in case_ids to mark for skip.
    :return: case_ids with added marks.
    """
    return _mark_cases(case_ids, xfails, pytest.mark.skip)


def filter(case_ids, to_filter):
    """Filter cases.

    :param case_ids: Parameter list for pytest.mark.parametrize.
    :param to_filter: List of parameters to filter from case_ids.
    :return: Filtered case_ids.
    """
    return [case for case in case_ids if case not in to_filter]
