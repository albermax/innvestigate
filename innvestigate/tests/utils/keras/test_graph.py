# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.models
import pytest


from innvestigate.utils.keras import graph as kgraph
from innvestigate.utils.tests import networks


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph():

    network_filter = "trivia.*:mnist.log_reg"

    for network in networks.iterator(network_filter):

        model = tensorflow.keras.models.Model(inputs=network["in"],
                                   outputs=network["out"])

        graph = kgraph.get_model_execution_graph(model)
        kgraph.print_model_execution_graph(graph)


@pytest.mark.precommit
def test_commit__get_model_execution_graph():

    network_filter = "mnist.*"

    for network in networks.iterator(network_filter):

        model = tensorflow.keras.models.Model(inputs=network["in"],
                                   outputs=network["out"])

        graph = kgraph.get_model_execution_graph(model)
        kgraph.print_model_execution_graph(graph)


@pytest.mark.precommit
def test_precommit__get_model_execution_graph_resnet50():

    network_filter = "imagenet.resnet50"

    for network in networks.iterator(network_filter):

        model = tensorflow.keras.models.Model(inputs=network["in"],
                                   outputs=network["out"])

        graph = kgraph.get_model_execution_graph(model)
        kgraph.print_model_execution_graph(graph)


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__get_model_execution_graph_with_inputs():

    network_filter = "trivia.*:mnist.log_reg"

    for network in networks.iterator(network_filter):

        model = tensorflow.keras.models.Model(inputs=network["in"],
                                   outputs=network["out"])

        graph = kgraph.get_model_execution_graph(model,
                                                 keep_input_layers=True)
        kgraph.print_model_execution_graph(graph)
