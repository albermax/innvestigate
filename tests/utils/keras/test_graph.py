from __future__ import annotations

import pytest

import innvestigate.utils.keras.graph as igraph

from tests import networks


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph():

    network_filter = "trivia.*:mnist.log_reg"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.mnist
@pytest.mark.precommit
def test_commit__get_model_execution_graph():

    network_filter = "mnist.*"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.resnet50
@pytest.mark.precommit
def test_precommit__get_model_execution_graph_resnet50():

    network_filter = "imagenet.resnet50"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph_with_inputs():

    network_filter = "trivia.*:mnist.log_reg"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model, keep_input_layers=True)
        igraph.print_model_execution_graph(graph)
