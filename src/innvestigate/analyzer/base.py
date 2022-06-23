from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import tensorflow.keras.models as kmodels

import innvestigate.analyzer
import innvestigate.backend.graph as igraph
import innvestigate.utils.sequence as isequence
from innvestigate.backend.types import LayerCheck, Model, ModelCheckDict, OptionalList

__all__ = [
    "NotAnalyzeableModelException",
    "AnalyzerBase",
    "TrainerMixin",
    "OneEpochTrainerMixin",
]


class NotAnalyzeableModelException(Exception):
    """Indicates that the model cannot be analyzed by an analyzer."""


class AnalyzerBase(metaclass=ABCMeta):
    """The basic interface of an iNNvestigate analyzer.

    This class defines the basic interface for analyzers:

    >>> model = create_keras_model()
    >>> a = Analyzer(model)
    >>> a.fit(X_train)  # If analyzer needs training.
    >>> analysis = a.analyze(X_test)
    >>>
    >>> state = a.save()
    >>> a_new = A.load(*state)
    >>> analysis = a_new.analyze(X_test)

    :param model: A Keras model.
    :param disable_model_checks: Do not execute model checks that enforce
      compatibility of analyzer and model.
    :param neuron_selection_mode: How to select the neuron to analyze.
      Possible values are 'max_activation', 'index' for the neuron
      (expects indices at :func:`analyze` calls), 'all' take all neurons.

    .. note:: To develop a new analyzer derive from
      :class:`AnalyzerNetworkBase`.
    """

    def __init__(
        self,
        model: Model,
        neuron_selection_mode: str = "max_activation",
        disable_model_checks: bool = False,
        _model_check_done: bool = False,
        _model_checks: list[ModelCheckDict] = None,
    ) -> None:
        self._model = model
        self._disable_model_checks = disable_model_checks
        self._model_check_done = _model_check_done

        # There are three possible neuron selection modes
        # that return an explanation w.r.t.:
        # * "max_activation": maximum activated neuron
        # * "index": neuron at index given on call to `analyze`
        # * "all": all output neurons
        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection_mode parameter is not valid.")
        self._neuron_selection_mode: str = neuron_selection_mode

        # If no model checks are given, initialize an empty list of checks
        # that child analyzers can append to.
        if _model_checks is None:
            _model_checks = []
        self._model_checks: list[ModelCheckDict] = _model_checks

    def _add_model_check(
        self, check: LayerCheck, message: str, check_type: str = "exception"
    ) -> None:
        """Add model check to list of checks `self._model_checks`.

        :param check: Callable that performs a boolean check on a Keras layers.
        :type check: LayerCheck
        :param message: Error message if check fails.
        :type message: str
        :param check_type: Either "exception" or "warning". Defaults to "exception"
        :type check_type: str, optional
        :raises Exception: [description]
        """

        if self._model_check_done:
            raise Exception(
                "Cannot add model check anymore. Check was already performed."
            )

        check_instance: ModelCheckDict = {
            "check": check,
            "message": message,
            "check_type": check_type,
        }
        self._model_checks.append(check_instance)

    def _do_model_checks(self) -> None:
        if not self._disable_model_checks and len(self._model_checks) > 0:
            check = [x["check"] for x in self._model_checks]
            types = [x["check_type"] for x in self._model_checks]
            messages = [x["message"] for x in self._model_checks]

            checked = igraph.model_contains(self._model, check)

            tmp = zip(checked, messages, types)

            for checked_layers, message, check_type in tmp:
                if len(checked_layers) > 0:
                    tmp_message = (
                        message + f"\nCheck triggered by layers: {checked_layers}"
                    )

                    if check_type == "exception":
                        raise NotAnalyzeableModelException(tmp_message)
                    if check_type == "warning":
                        # TODO(albermax) only the first warning will be shown
                        warnings.warn(tmp_message)
                    raise NotImplementedError()

        self._model_check_done = True

    def fit(self, *_args, disable_no_training_warning: bool = False, **_kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        if not disable_no_training_warning:
            # issue warning if no training is foreseen, but fit() is still called.
            warnings.warn(
                "This analyzer does not need to be trained." " Still fit() is called.",
                RuntimeWarning,
            )

    def fit_generator(
        self, *_args, disable_no_training_warning: bool = False, **_kwargs
    ):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        if not disable_no_training_warning:
            # issue warning if no training is foreseen, but fit() is still called.
            warnings.warn(
                "This analyzer does not need to be trained."
                " Still fit_generator() is called.",
                RuntimeWarning,
            )

    @abstractmethod
    def analyze(
        self, X: OptionalList[np.ndarray], *args: Any, **kwargs: Any
    ) -> OptionalList[np.ndarray]:
        """
        Analyze the behavior of model on input `X`.

        :param X: Input as expected by model.
        """

    def _get_state(self) -> dict:
        state = {
            "model_json": self._model.to_json(),
            "model_weights": self._model.get_weights(),
            "disable_model_checks": self._disable_model_checks,
            "neuron_selection_mode": self._neuron_selection_mode,
        }
        return state

    def save(self) -> tuple[str, dict]:
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load`
        to resemble the analyzer.

        :return: The class name and the state.
        """
        state = self._get_state()
        class_name = self.__class__.__name__
        return class_name, state

    def save_npz(self, fname: str) -> None:
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load_npz`
        to resemble the analyzer.

        :param fname: The file's name.
        """
        class_name, state = self.save()
        np.savez(fname, **{"class_name": class_name, "state": state})

    @classmethod
    def _state_to_kwargs(cls, state: dict) -> dict:
        disable_model_checks = state.pop("disable_model_checks")
        model_json = state.pop("model_json")
        model_weights = state.pop("model_weights")
        neuron_selection_mode = state.pop("neuron_selection_mode")

        # since `super()._state_to_kwargs(state)` should be called last
        # in every child class, the dict `state` should be empty at this point.
        if len(state) != 0:
            raise RuntimeError(f"Serialization failed. Got left-over state {state}.")

        model = kmodels.model_from_json(model_json)
        model.set_weights(model_weights)
        return {
            "model": model,
            "disable_model_checks": disable_model_checks,
            "neuron_selection_mode": neuron_selection_mode,
        }

    @staticmethod
    def load(class_name: str, state: dict[str, Any]) -> AnalyzerBase:
        """
        Resembles an analyzer from the state created by
        :func:`analyzer.save()`.

        :param class_name: The analyzer's class name.
        :param state: The analyzer's state.
        """
        # TODO: do in a smarter way without cyclic imports
        cls = getattr(innvestigate.analyzer, class_name)

        kwargs = cls._state_to_kwargs(state)
        return cls(**kwargs)  # type: ignore

    @staticmethod
    def load_npz(fname):
        """
        Resembles an analyzer from the file created by
        :func:`analyzer.save_npz()`.

        :param fname: The file's name.
        """
        npz_file = np.load(fname)

        class_name = npz_file["class_name"].item()
        state = npz_file["state"].item()
        return AnalyzerBase.load(class_name, state)


###############################################################################


class TrainerMixin:
    """Mixin for analyzer that adapt to data.

    This convenience interface exposes a Keras like training routing
    to the user.
    """

    # TODO: extend with Y
    def fit(self, X: np.ndarray | None = None, batch_size: int = 32, **kwargs) -> None:
        """
        Takes the same parameters as Keras's :func:`model.fit` function.
        """
        generator = isequence.BatchSequence(X, batch_size)
        return self._fit_generator(generator, **kwargs)  # type: ignore

    def fit_generator(self, *args, **kwargs):
        """
        Takes the same parameters as Keras's :func:`model.fit_generator`
        function.
        """
        return self._fit_generator(*args, **kwargs)

    def _fit_generator(self, *_args, **_kwargs):
        raise NotImplementedError()


class OneEpochTrainerMixin(TrainerMixin):
    """Exposes the same interface and functionality as :class:`TrainerMixin`
    except that the training is limited to one epoch.
    """

    def fit(self, *args, **kwargs) -> None:
        """
        Same interface as :func:`fit` of :class:`TrainerMixin` except that
        the parameter epoch is fixed to 1.
        """
        return super().fit(*args, epochs=1, **kwargs)

    def fit_generator(self, *args, steps: int = None, **kwargs):
        """
        Same interface as :func:`fit_generator` of :class:`TrainerMixin` except that
        the parameter epoch is fixed to 1.
        """
        return super().fit_generator(*args, steps_per_epoch=steps, epochs=1, **kwargs)
