"""Iterate over test networks specified by filter string."""
from __future__ import annotations

import fnmatch
import types
from typing import Iterator

import tensorflow.keras.backend as kbackend

from innvestigate.backend.types import Model

from tests.networks import imagenet, mnist, trivia


def iterator(
    network_filter: str = "*", clear_sessions: bool = False
) -> Iterator[Model]:
    """Iterator over various neural networks. Networks can be filtered by name.

    :param network_filter: Names of the networks that should be part of the iterator.
        Use delimiter `:` to separate names, `*` as wildcard. Defaults to `*`.
    :type network_filter: str, optional
    :param clear_sessions: Flag that clears Keras backend session
        before loading each network, defaults to False.
    :type clear_sessions: bool, optional
    """

    networks = (
        _fetch_networks("trivia", trivia, network_filter)
        + _fetch_networks("mnist", mnist, network_filter)
        + _fetch_networks("imagenet", imagenet, network_filter)
    )

    for _module_name, (module, name) in networks:
        if clear_sessions:
            kbackend.clear_session()
        network: Model = getattr(module, name)()
        yield network


def _fetch_networks(
    module_name: str, module, network_filter: str
) -> list[tuple[str, tuple[types.ModuleType, str]]]:
    """Fetch all networks in the given module that match the network_filter string."""
    networks = [
        (f"{module_name}.{name}", (module, name))
        for name in module.__all__
        if any(
            (
                fnmatch.fnmatch(name, one_filter)
                or fnmatch.fnmatch(f"{module_name}.{name}", one_filter)
            )
            for one_filter in network_filter.split(":")
        )
    ]
    return [n for n in sorted(networks)]
