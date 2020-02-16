# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate import backend
from innvestigate.utils.keras import graph as kgraph
from innvestigate.utils.tests import cases


require_tf = pytest.mark.skipif(backend.name() != "tensorflow",
                                reason="Testing TF only functionality.")


###############################################################################
###############################################################################
###############################################################################


@require_tf
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST+cases.PRECOMMIT)
def test_fast__get_model_execution_graph(case_id):

    case = getattr(cases, case_id)
    if case is None:
        raise ValueError("Invalid case_id.")

    model, _ = case()

    graph = kgraph.get_model_execution_graph(model)
    kgraph.print_model_execution_graph(graph)


@require_tf
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST+cases.PRECOMMIT)
def test_fast__get_model_execution_graph_with_inputs(case_id):

    case = getattr(cases, case_id)
    if case is None:
        raise ValueError("Invalid case_id.")

    model, _ = case()

    graph = kgraph.get_model_execution_graph(model,
                                             keep_input_layers=True)
    kgraph.print_model_execution_graph(graph)
