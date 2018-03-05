# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import keras.layers
import keras.models
import numpy as np
import warnings


from .. import layers as ilayers
from .. import utils as iutils
from ..utils.keras import graph as kgraph


__all__ = [
    "AnalyzerBase",

    "TrainerMixin",
    "OneEpochTrainerMixin",

    "AnalyzerNetworkBase",
    "ReverseAnalyzerBase"
]


###############################################################################
###############################################################################
###############################################################################


class AnalyzerBase(object):
    """
    The basic interface of an Innvestigate analyzer.
    """

    # Should be specified by the base class.
    _model_checks = []

    def __init__(self, model, model_checks_raise_exception=True):
        self._model = model
        self._model_checks_raise_exception = model_checks_raise_exception

        if len(self._model_checks) > 0:
            checks = [x[0] for x in self._model_checks]
            messages = [x[1] for x in self._model_checks]
            checked = kgraph.model_contains(self._model, checks,
                                            return_only_counts=True)
            for check_count, message in zip(iutils.listify(checked), messages):
                if check_count > 0:
                    if self._model_checks_raise_exception is True:
                        raise Exception(message)
                    else:
                        warnings.warn(message)
        pass

    def fit(self, *args, disable_no_training_warning=False, **kwargs):
        if not disable_no_training_warning:
            # issue warning if not training is foreseen,
            # but is fit is still called.
            warnings.warn("This analyzer does not need to be trained."
                          " Still fit() is called.", RuntimeWarning)
        pass

    def fit_generator(self, *args,
                      disable_no_training_warning=False, **kwargs):
        if not disable_no_training_warning:
            # issue warning if not training is foreseen,
            # but is fit is still called.
            warnings.warn("This analyzer does not need to be trained."
                          " Still fit_generator() is called.", RuntimeWarning)
        pass

    def analyze(self, X):
        raise NotImplementedError()

    def _get_state(self):
        model_json = self._model.to_json()
        model_weights = self._model.get_weights()
        return {"model_json": model_json, "model_weights": model_weights}

    def save(self):
        state = self._get_state()
        class_name = self.__class__.__name__
        return class_name, state

    def save_npz(self, fname):
        class_name, state = self.save()
        np.savez(fname, **{"class_name": class_name,
                           "state": state})
        pass

    @classmethod
    def _state_to_kwargs(clazz, state):
        model_json = state.pop("model_json")
        model_weights = state.pop("model_weights")
        assert len(state) == 0

        model = keras.models.model_from_json(model_json)
        model.set_weights(model_weights)
        return {"model": model}

    @staticmethod
    def load(class_name, state):
        # Todo:do in a smarter way!
        import innvestigate.analyzer
        clazz = getattr(innvestigate.analyzer, class_name)

        kwargs = clazz._state_to_kwargs(state)
        return clazz(**kwargs)

    @staticmethod
    def load_npz(fname):
        f = np.load(fname)

        class_name = f["class_name"].item()
        state = f["state"].item()
        return AnalyzerBase.load(class_name, state)


###############################################################################
###############################################################################
###############################################################################


class TrainerMixin(object):

    # todo: extend with Y
    def fit(self,
            X=None,
            batch_size=32,
            **kwargs):
        generator = iutils.BatchSequence(X, batch_size)
        return self._fit_generator(generator,
                                  **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self._fit_generator(*args, **kwargs)

    def _fit_generator(self,
                       generator,
                       steps_per_epoch=None,
                       epochs=1,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0,
                       disable_no_training_warning=None):
        raise NotImplementedError()
    pass


class OneEpochTrainerMixin(TrainerMixin):

    def fit(self, *args, **kwargs):
        return super(OneEpochTrainerMixin, self).fit(*args, epochs=1, **kwargs)

    def fit_generator(self, *args, steps=None, **kwargs):
        return super(OneEpochTrainerMixin, self).fit_generator(
            *args,
            steps_per_epoch=steps,
            epochs=1,
            **kwargs)


###############################################################################
###############################################################################
###############################################################################


class AnalyzerNetworkBase(AnalyzerBase):
    """
    Analyzer itself is defined as keras graph.
    """

    def __init__(self, model, neuron_selection_mode="max_activation"):
        super(AnalyzerNetworkBase, self).__init__(model)

        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection parameter is not valid.")
        self._neuron_selection_mode = neuron_selection_mode
        pass

    def compile_analyzer(self):
        model = self._model
        neuron_selection_mode = self._neuron_selection_mode

        neuron_selection_inputs = []
        model_inputs, model_output = model.inputs, model.outputs
        if len(model_output) > 1:
            raise ValueError("Only models with one output tensor are allowed.")

        if neuron_selection_mode == "max_activation":
            model_output = ilayers.Max()(model_output)
        if neuron_selection_mode == "index":
            # todo: implement index mode
            raise NotImplementedError("Only a stub present so far.")
            neuron_indexing = keras.layers.Input(shape=[None, None])
            neuron_selection_inputs += neuron_indexing

            model_output = keras.layers.Index()([model_output, neuron_indexing])

        model = keras.models.Model(inputs=model_inputs+neuron_selection_inputs,
                                   outputs=model_output)
        tmp = self._create_analysis(model)
        try:
            analysis_outputs, debug_outputs, constant_inputs = tmp
        except (TypeError, ValueError):
            try:
                analysis_outputs, debug_outputs = tmp
                constant_inputs = list()
            except (TypeError, ValueError):
                analysis_outputs = iutils.listify(tmp)
                constant_inputs, debug_outputs = list(), list()

        self._n_data_input = len(model_inputs)
        self._n_constant_input = len(constant_inputs)
        self._n_data_output = len(analysis_outputs)
        self._n_debug_output = len(debug_outputs)
        self._analyzer_model = keras.models.Model(
            inputs=model_inputs+neuron_selection_inputs+constant_inputs,
            outputs=analysis_outputs+debug_outputs)
        #self._analyzer_model.compile(optimizer="sgd", loss="mse")
        pass

    def _create_analysis(self, model):
        raise NotImplementedError()

    def analyze(self, X, neuron_selection=None):
        if not hasattr(self, "_analyzer_model"):
            self.compile_analyzer()

        # todo: update all interfaces, X can be a list.
        if(neuron_selection is not None and
           self._neuron_selection_mode != "index"):
            raise ValueError("Only neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")
        if(neuron_selection is None and
           self._neuron_selection_mode == "index"):
            raise ValueError("neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")

        if self._neuron_selection_mode == "index":
            ret = self._analyzer_model.predict_on_batch(X, neuron_selection)
        else:
            ret = self._analyzer_model.predict_on_batch(X)

        if self._n_debug_output > 0:
            self._handle_debug_output(ret[-self._n_debug_output:])
            ret = ret[:-self._n_debug_output]

        if isinstance(ret, list) and len(ret) == 1:
            ret = ret[0]
        return ret

    def _get_state(self):
        state = super(AnalyzerNetworkBase, self)._get_state()
        state.update({"neuron_selection_mode": self._neuron_selection_mode})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        neuron_selection_mode = state.pop("neuron_selection_mode")
        kwargs = super(AnalyzerNetworkBase, clazz)._state_to_kwargs(state)
        kwargs.update({"neuron_selection_mode": neuron_selection_mode})
        return kwargs


class ReverseAnalyzerBase(AnalyzerNetworkBase):

    # Should be specified by the base class.
    _conditional_mappings = []

    def __init__(self, *args,
                 reverse_verbose=False,
                 reverse_clip_values=False,
                 reverse_check_min_max_values=False,
                 reverse_check_finite=False,
                 reverse_reapply_on_copied_layers=False,
                 **kwargs):
        self._reverse_verbose = reverse_verbose
        self._reverse_clip_values = reverse_clip_values
        self._reverse_check_min_max_values = reverse_check_min_max_values
        self._reverse_check_finite = reverse_check_finite
        self._reverse_reapply_on_copied_layers = (
            reverse_reapply_on_copied_layers)
        return super(ReverseAnalyzerBase, self).__init__(*args, **kwargs)

    def _reverse_mapping(self, layer):
        for condition, reverse_f in self._conditional_mappings:
            if condition(layer):
                return reverse_f
        return None

    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        # The gradient.
        return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

    def _head_mapping(self, X):
        return X

    def _create_analysis(self, model):
        return_all_reversed_tensors = (
            self._reverse_check_min_max_values or
            self._reverse_check_finite
        )
        ret = kgraph.reverse_model(
            model,
            reverse_mappings=self._reverse_mapping,
            default_reverse_mapping=self._default_reverse_mapping,
            head_mapping=self._head_mapping,
            verbose=self._reverse_verbose,
            clip_all_reversed_tensors=self._reverse_clip_values,
            return_all_reversed_tensors=return_all_reversed_tensors)

        if return_all_reversed_tensors:
            debug_tensors = []
            self._debug_tensors_indices = {}

            values = list(six.itervalues(ret[1]))
            mapping = {i: v["id"] for i, v in enumerate(values)}
            tensors = [v["tensor"] for v in values]
            self._reverse_tensors_mapping = mapping

            if self._reverse_check_min_max_values:
                tmp = [ilayers.Min(None)(x) for x in tensors]
                self._debug_tensors_indices["min"] = (
                    len(debug_tensors),
                    len(debug_tensors)+len(tmp))
                debug_tensors += tmp

                tmp = [ilayers.Max(None)(x) for x in tensors]
                self._debug_tensors_indices["max"] = (
                    len(debug_tensors),
                    len(debug_tensors)+len(tmp))
                debug_tensors += tmp

            if self._reverse_check_finite:
                tmp = iutils.listify(ilayers.FiniteCheck()(tensors))
                self._debug_tensors_indices["finite"] = (
                    len(debug_tensors),
                    len(debug_tensors)+len(tmp))
                debug_tensors += tmp

            ret = (ret[0], debug_tensors)
        return ret

    def _handle_debug_output(self, debug_values):

        if self._reverse_check_min_max_values:
            indices = self._debug_tensors_indices["min"]
            tmp = debug_values[indices[0]:indices[1]]
            tmp = sorted([(self._reverse_tensors_mapping[i], v)
                          for i, v in enumerate(tmp)])
            print("Minimum values in tensors: "
                  "((ReverseID, TensorID), Value) - {}".format(tmp))

            indices = self._debug_tensors_indices["max"]
            tmp = debug_values[indices[0]:indices[1]]
            tmp = sorted([(self._reverse_tensors_mapping[i], v)
                          for i, v in enumerate(tmp)])
            print("Maximum values in tensors: "
                  "((ReverseID, TensorID), Value) - {}".format(tmp))

        if self._reverse_check_finite:
            indices = self._debug_tensors_indices["finite"]
            tmp = debug_values[indices[0]:indices[1]]
            nfinite_tensors = np.flatnonzero(np.asarray(tmp) > 0)

            if len(nfinite_tensors) > 0:
                nfinite_tensors = sorted([self._reverse_tensors_mapping[i]
                                          for i in nfinite_tensors])
                print("Not finite values found in following nodes: "
                      "(ReverseID, TensorID) - {}".format(nfinite_tensors))
        pass

    def _get_state(self):
        state = super(ReverseAnalyzerBase, self)._get_state()
        state.update({"reverse_verbose": self._reverse_verbose})
        state.update({"reverse_check_finite": self._reverse_check_finite})
        state.update({"reverse_reapply_on_copied_layers":
                      self._reverse_reapply_on_copied_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        reverse_verbose = state.pop("reverse_verbose")
        reverse_check_finite = state.pop("reverse_check_finite")
        reverse_reapply_on_copied_layers = (
            state.pop("reverse_reapply_on_copied_layers"))
        kwargs = super(ReverseAnalyzerBase, clazz)._state_to_kwargs(state)
        kwargs.update({"reverse_verbose": reverse_verbose,
                       "reverse_check_finite": reverse_check_finite,
                       "reverse_reapply_on_copied_layers":
                       reverse_reapply_on_copied_layers})
        return kwargs
