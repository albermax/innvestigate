from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow.keras.backend as kbackend

import innvestigate.backend as ibackend
import innvestigate.backend.graph as igraph
import innvestigate.layers as ilayers
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.backend.types import (
    CondReverseMapping,
    Layer,
    Model,
    OptionalList,
    ReverseTensorDict,
    Tensor,
)

__all__ = ["ReverseAnalyzerBase"]


class ReverseAnalyzerBase(AnalyzerNetworkBase):
    """Convenience class for analyzers that revert the model's structure.

    This class contains many helper functions around the graph
    reverse function :func:`innvestigate.backend.graph.reverse_model`.

    The deriving classes should specify how the graph should be reverted
    by implementing the following functions:

    * :func:`_reverse_mapping(layer)` given a layer this function
      returns a reverse mapping for the layer as specified in
      :func:`innvestigate.backend.graph.reverse_model` or None.

      This function can be implemented, but it is encouraged to
      implement a default mapping and add additional changes with
      the function :func:`_add_conditional_reverse_mapping` (see below).

      The default behavior is finding a conditional mapping (see below),
      if none is found, :func:`_default_reverse_mapping` is applied.
    * :func:`_default_reverse_mapping` defines the default
      reverse mapping.
    * :func:`_head_mapping` defines how the outputs of the model
      should be instantiated before the are passed to the reversed
      network.

    Furthermore other parameters of the function
    :func:`innvestigate.backend.graph.reverse_model` can
    be changed by setting the according parameters of the
    init function:

    :param reverse_verbose: Print information on the reverse process.
    :param reverse_clip_values: Clip the values that are passed along
      the reverted network. Expects tuple (min, max).
    :param reverse_project_bottleneck_layers: Project the value range
      of bottleneck tensors in the reverse network into another range.
    :param reverse_check_min_max_values: Print the min/max values
      observed in each tensor along the reverse network whenever
      :func:`analyze` is called.
    :param reverse_check_finite: Check if values passed along the
      reverse network are finite.
    :param reverse_keep_tensors: Keeps the tensors created in the
      backward pass and stores them in the attribute
      :attr:`_reversed_tensors`.
    :param reverse_reapply_on_copied_layers: See
      :func:`innvestigate.backend.graph.reverse_model`.
    """

    def __init__(
        self,
        model: Model,
        reverse_verbose: bool = False,
        reverse_clip_values: bool = False,
        reverse_project_bottleneck_layers: bool = False,
        reverse_check_min_max_values: bool = False,
        reverse_check_finite: bool = False,
        reverse_keep_tensors: bool = False,
        reverse_reapply_on_copied_layers: bool = False,
        **kwargs,
    ) -> None:
        """
        From AnalyzerBase super init:
        * Initializes empty list of _model_checks

        From AnalyzerNetworkBase super init:
        * set _neuron_selection_mode
        * add check for lambda layers through 'allow_lambda_layers'
        * define attributes for '_prepare_model', which is later called
            through 'create_analyzer_model'

        Here:
        * define attributes required for calling '_conditional_reverse_mapping'
        """
        super().__init__(model, **kwargs)

        self._reverse_verbose = reverse_verbose
        self._reverse_clip_values = reverse_clip_values
        self._reverse_project_bottleneck_layers = reverse_project_bottleneck_layers
        self._reverse_check_min_max_values = reverse_check_min_max_values
        self._reverse_check_finite = reverse_check_finite
        self._reverse_keep_tensors = reverse_keep_tensors
        self._reverse_reapply_on_copied_layers = reverse_reapply_on_copied_layers
        self._reverse_mapping_applied: bool = False

        # map priorities to lists of conditional reverse mappings
        self._conditional_reverse_mappings: dict[int, list[CondReverseMapping]] = {}

        # Maps keys "min", "max", "finite", "keep" to tuples of indices
        self._debug_tensors_indices: dict[str, tuple[int, int]] = {}

    def _gradient_reverse_mapping(
        self,
        Xs: OptionalList[Tensor],
        Ys: OptionalList[Tensor],
        reversed_Ys: OptionalList[Tensor],
        reverse_state: dict,
    ):
        """Returns masked gradient."""
        mask = [id(X) not in reverse_state["stop_mapping_at_ids"] for X in Xs]
        grad = ibackend.gradients(Xs, Ys, reversed_Ys)
        return ibackend.apply_mask(grad, mask)

    def _reverse_mapping(self, layer: Layer):
        """
        This function should return a reverse mapping for the passed layer.

        If this function returns None, :func:`_default_reverse_mapping`
        is applied.

        :param layer: The layer for which a mapping should be returned.
        :return: The mapping can be of the following forms:
          * A function of form (A) f(Xs, Ys, reversed_Ys, reverse_state)
            that maps reversed_Ys to reversed_Xs (which should contain
            tensors of the same shape and type).
          * A function of form f(B) f(layer, reverse_state) that returns
            a function of form (A).
          * A :class:`ReverseMappingBase` subclass.
        """
        if layer in self._special_helper_layers:
            # Special layers added by AnalyzerNetworkBase
            # that should not be exposed to user.
            return self._gradient_reverse_mapping

        return self._apply_conditional_reverse_mappings(layer)

    def _add_conditional_reverse_mapping(
        self,
        condition: Callable[[Layer], bool],
        mapping: Callable,  # TODO: specify type of Callable
        priority: int = -1,
        name: str | None = None,
    ):
        """
        This function should return a reverse mapping for the passed layer.

        If this function returns None, :func:`_default_reverse_mapping`
        is applied.

        :param condition: Condition when this mapping should be applied.
          Form: f(layer) -> bool
        :param mapping: The mapping can be of the following forms:
          * A function of form (A) f(Xs, Ys, reversed_Ys, reverse_state)
            that maps reversed_Ys to reversed_Xs (which should contain
            tensors of the same shape and type).
          * A function of form f(B) f(layer, reverse_state) that returns
            a function of form (A).
          * A :class:`ReverseMappingBase` subclass.
        :param priority: The higher the earlier the condition gets
          evaluated.
        :param name: An identifying name.
        """
        if self._reverse_mapping_applied is True:
            raise Exception(
                "Cannot add conditional mapping " "after first application."
            )

        # Add key `priority` to dict _conditional_reverse_mappings
        # if it doesn't exist yet.
        if priority not in self._conditional_reverse_mappings:
            self._conditional_reverse_mappings[priority] = []

        # Add Conditional Reveserse mapping at given priority
        tmp: CondReverseMapping = {
            "condition": condition,
            "mapping": mapping,
            "name": name,
        }
        self._conditional_reverse_mappings[priority].append(tmp)

    def _apply_conditional_reverse_mappings(self, layer):
        mappings = getattr(self, "_conditional_reverse_mappings", {})
        self._reverse_mapping_applied = True

        # Search for mapping. First consider ones with highest priority,
        # inside priority in order of adding.
        sorted_keys = reversed(sorted(mappings.keys()))
        for key in sorted_keys:
            for mapping in mappings[key]:
                if mapping["condition"](layer):
                    return mapping["mapping"]

        # Otherwise return None and default reverse mapping will be applied
        return None

    def _default_reverse_mapping(
        self,
        Xs: OptionalList[Tensor],
        Ys: OptionalList[Tensor],
        reversed_Ys: OptionalList[Tensor],
        reverse_state: dict,
    ):
        """
        Fallback function to map reversed_Ys to reversed_Xs
        (which should contain tensors of the same shape and type).
        """
        return self._gradient_reverse_mapping(Xs, Ys, reversed_Ys, reverse_state)

    def _head_mapping(self, X: Tensor) -> Tensor:
        """
        Map output tensors to new values before passing
        them into the reverted network.
        """
        # Here: Keep the output signal.
        # Should be re-implemented by inheritance.
        # Refer to the "Introduction to development notebook".
        return X

    def _postprocess_analysis(self, Xs: OptionalList[Tensor]) -> list[Tensor]:
        return ibackend.to_list(Xs)

    def _reverse_model(
        self,
        model: Model,
        stop_analysis_at_tensors: list[Tensor] = None,
        return_all_reversed_tensors=False,
    ) -> tuple[list[Tensor], dict[Tensor, ReverseTensorDict] | None]:
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        return igraph.reverse_model(
            model,
            reverse_mappings=self._reverse_mapping,
            default_reverse_mapping=self._default_reverse_mapping,
            head_mapping=self._head_mapping,
            stop_mapping_at_tensors=stop_analysis_at_tensors,
            verbose=self._reverse_verbose,
            clip_all_reversed_tensors=self._reverse_clip_values,
            project_bottleneck_tensors=self._reverse_project_bottleneck_layers,
            return_all_reversed_tensors=return_all_reversed_tensors,
        )

    def _create_analysis(
        self, model: Model, stop_analysis_at_tensors: list[Tensor] = None
    ):

        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        return_all_reversed_tensors = (
            self._reverse_check_min_max_values
            or self._reverse_check_finite
            or self._reverse_keep_tensors
        )

        # if return_all_reversed_tensors is False,
        # reversed_tensors will be None
        reversed_input_tensors, reversed_tensors = self._reverse_model(
            model,
            stop_analysis_at_tensors=stop_analysis_at_tensors,
            return_all_reversed_tensors=return_all_reversed_tensors,
        )
        ret = self._postprocess_analysis(reversed_input_tensors)

        if return_all_reversed_tensors:
            if reversed_tensors is None:
                raise TypeError("Expected reversed_tensors, got None.")

            debug_tensors: list[Tensor]
            tmp: list[Tensor]

            debug_tensors = []
            values = reversed_tensors.values()
            mapping = {i: v["nid"] for i, v in enumerate(values)}
            tensors = [v["final_tensor"] for v in values]
            self._reverse_tensors_mapping = mapping

            if self._reverse_check_min_max_values:
                tmp = [kbackend.min(x) for x in tensors]
                self._debug_tensors_indices["min"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp),
                )
                debug_tensors += tmp

                tmp = [kbackend.min(x) for x in tensors]
                self._debug_tensors_indices["max"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp),
                )
                debug_tensors += tmp

            if self._reverse_check_finite:
                tmp = ibackend.to_list(ilayers.FiniteCheck()(tensors))
                self._debug_tensors_indices["finite"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp),
                )
                debug_tensors += tmp

            if self._reverse_keep_tensors:
                self._debug_tensors_indices["keep"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tensors),
                )
                debug_tensors += tensors

            return ret, debug_tensors
        return ret

    def _handle_debug_output(self, debug_values):

        if self._reverse_check_min_max_values:
            indices = self._debug_tensors_indices["min"]
            tmp = debug_values[indices[0] : indices[1]]
            tmp = sorted(
                (self._reverse_tensors_mapping[i], v) for i, v in enumerate(tmp)
            )
            print(f"Minimum values in tensors: ((NodeID, TensorID), Value) - {tmp}")

            indices = self._debug_tensors_indices["max"]
            tmp = debug_values[indices[0] : indices[1]]
            tmp = sorted(
                (self._reverse_tensors_mapping[i], v) for i, v in enumerate(tmp)
            )
            print(f"Maximum values in tensors: ((NodeID, TensorID), Value) - {tmp}")

        if self._reverse_check_finite:
            indices = self._debug_tensors_indices["finite"]
            tmp = debug_values[indices[0] : indices[1]]
            nfinite_tensors = np.flatnonzero(np.asarray(tmp) > 0)

            if len(nfinite_tensors) > 0:
                nfinite_tensors = sorted(
                    self._reverse_tensors_mapping[i] for i in nfinite_tensors
                )
                print(
                    "Not finite values found in following nodes: "
                    f"(NodeID, TensorID) - {nfinite_tensors}"
                )

        if self._reverse_keep_tensors:
            indices = self._debug_tensors_indices["keep"]
            tmp = debug_values[indices[0] : indices[1]]
            tmp = sorted(
                (self._reverse_tensors_mapping[i], v) for i, v in enumerate(tmp)
            )
            self._reversed_tensors = tmp

    def _get_state(self):
        state = super()._get_state()
        state.update(
            {
                "reverse_verbose": self._reverse_verbose,
                "reverse_clip_values": self._reverse_clip_values,
                "reverse_project_bottleneck_layers": self._reverse_project_bottleneck_layers,  # noqa
                "reverse_check_min_max_values": self._reverse_check_min_max_values,
                "reverse_check_finite": self._reverse_check_finite,
                "reverse_keep_tensors": self._reverse_keep_tensors,
                "reverse_reapply_on_copied_layers": self._reverse_reapply_on_copied_layers,  # noqa
            }
        )
        return state

    @classmethod
    def _state_to_kwargs(cls, state: dict):
        reverse_verbose = state.pop("reverse_verbose")
        reverse_clip_values = state.pop("reverse_clip_values")
        reverse_project_bottleneck_layers = state.pop(
            "reverse_project_bottleneck_layers"
        )
        reverse_check_min_max_values = state.pop("reverse_check_min_max_values")
        reverse_check_finite = state.pop("reverse_check_finite")
        reverse_keep_tensors = state.pop("reverse_keep_tensors")
        reverse_reapply_on_copied_layers = state.pop("reverse_reapply_on_copied_layers")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update(
            {
                "reverse_verbose": reverse_verbose,
                "reverse_clip_values": reverse_clip_values,
                "reverse_project_bottleneck_layers": reverse_project_bottleneck_layers,
                "reverse_check_min_max_values": reverse_check_min_max_values,
                "reverse_check_finite": reverse_check_finite,
                "reverse_keep_tensors": reverse_keep_tensors,
                "reverse_reapply_on_copied_layers": reverse_reapply_on_copied_layers,
            }
        )
        return kwargs
