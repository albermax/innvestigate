# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import numpy as np

from ...analyzer import NotAnalyzeableModelException
from ... import backend
from . import cases


__all__ = [
    "test_analyzer",
    "test_analyzers_for_same_output",
]


###############################################################################
###############################################################################
###############################################################################


def test_analyzer(case_id, create_analyzer_f):
    np.random.seed(2349784365)
    # Keras is present close tf session.
    if backend.K:
        backend.K.clear_session()

    # Fetch case.
    case = getattr(cases, case_id, None)
    if case is None:
        raise ValueError("Invalid case_id.")

    model, data = case()
    try:
        analyzer = create_analyzer_f(model)
    except NotAnalyzeableModelException:
        # Not being able to analyze is ok.
        return

    # Dryrun.
    analyzer.fit(data)
    analysis = analyzer.analyze(data)

    # Check if numbers are valid.
    assert analysis.shape == data.shape
    assert not np.any(np.isinf(analysis.ravel()))
    assert not np.any(np.isnan(analysis.ravel()))


def test_analyzers_for_same_output(
        case_id, create_analyzer1_f, create_analyzer2_f, rtol=None, atol=None):
    np.random.seed(2349784365)
    # Keras is present close tf session.
    if backend.K:
        backend.K.clear_session()

    # Fetch case.
    case = getattr(cases, case_id, None)
    if case is None:
        raise ValueError("Invalid case_id.")

    model, data = case()
    try:
        analyzer1 = create_analyzer1_f(model)
        analyzer2 = create_analyzer2_f(model)
    except NotAnalyzeableModelException:
        # Not being able to analyze is ok.
        return

    # Dryrun.
    analyzer1.fit(data)
    analysis1 = analyzer1.analyze(data)

    analyzer2.fit(data)
    analysis2 = analyzer2.analyze(data)

    # Check if numbers are valid.
    assert analysis1.shape == data.shape
    assert not np.any(np.isinf(analysis1.ravel()))
    assert not np.any(np.isnan(analysis1.ravel()))

    assert analysis2.shape == data.shape
    assert not np.any(np.isinf(analysis2.ravel()))
    assert not np.any(np.isnan(analysis2.ravel()))

    # Check if the results match.
    all_close_kwargs = {}
    if rtol:
        all_close_kwargs["rtol"] = rtol
    if atol:
        all_close_kwargs["atol"] = atol
    assert np.allclose(analysis1, analysis2, **all_close_kwargs)
