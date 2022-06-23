from __future__ import annotations

import pytest
import tensorflow as tf

import innvestigate.backend.graph as igraph

from tests import networks


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph():
    tf.keras.backend.clear_session()

    network_filter = "trivia.*:mnist.log_reg"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.graph
@pytest.mark.mnist
@pytest.mark.precommit
def test_commit__get_model_execution_graph():
    tf.keras.backend.clear_session()

    network_filter = "mnist.*"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.graph
@pytest.mark.resnet50
@pytest.mark.precommit
def test_precommit__get_model_execution_graph_resnet50():
    tf.keras.backend.clear_session()

    network_filter = "imagenet.resnet50"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph_with_inputs():
    tf.keras.backend.clear_session()

    network_filter = "trivia.*:mnist.log_reg"

    for model in networks.iterator(network_filter):
        graph = igraph.get_model_execution_graph(model, keep_input_layers=True)
        igraph.print_model_execution_graph(graph)
