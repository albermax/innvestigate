from __future__ import annotations

import pytest
import tensorflow.keras.models as kmodels

import innvestigate.utils.keras.graph as igraph

from tests import networks


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph():

    network_filter = "trivia.*:mnist.log_reg"

    for network in networks.iterator(network_filter):

        model = kmodels.Model(inputs=network["in"], outputs=network["out"])

        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.precommit
def test_commit__get_model_execution_graph():

    network_filter = "mnist.*"

    for network in networks.iterator(network_filter):

        model = kmodels.Model(inputs=network["in"], outputs=network["out"])

        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.precommit
def test_precommit__get_model_execution_graph_resnet50():

    network_filter = "imagenet.resnet50"

    for network in networks.iterator(network_filter):

        model = kmodels.Model(inputs=network["in"], outputs=network["out"])

        graph = igraph.get_model_execution_graph(model)
        igraph.print_model_execution_graph(graph)


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph_with_inputs():

    network_filter = "trivia.*:mnist.log_reg"

    for network in networks.iterator(network_filter):

        model = kmodels.Model(inputs=network["in"], outputs=network["out"])

        graph = igraph.get_model_execution_graph(model, keep_input_layers=True)
        igraph.print_model_execution_graph(graph)
