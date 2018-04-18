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
    """ The basic interface of an iNNvestigate analyzer.

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

    .. note:: To develop a new analyzer derive from
      :class:`AnalyzerNetworkBase`.
    """

    # Should be specified by the base class.
    _model_checks = []

    def __init__(self, model, disable_model_checks=False):
        self._model = model
        self._disable_model_checks = disable_model_checks

        if not self._disable_model_checks and len(self._model_checks) > 0:
            checks = [x["check"] for x in self._model_checks]
            types = [x.get("type", "exception") for x in self._model_checks]
            messages = [x["message"] for x in self._model_checks]

            checked = kgraph.model_contains(self._model, checks,
                                            return_only_counts=True)
            tmp = zip(iutils.to_list(checked), messages, types)
            for check_count, message, check_type in tmp:
                if check_count > 0:
                    if check_type == "exception":
                        raise Exception(message)
                    elif check_type == "warning":
                        # TODO: fix only first warning is showed.
                        # but all should be.
                        warnings.warn(message)
                    else:
                        raise NotImplementedError()

    def fit(self, *args, disable_no_training_warning=False, **kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        if not disable_no_training_warning:
            # issue warning if not training is foreseen,
            # but is fit is still called.
            warnings.warn("This analyzer does not need to be trained."
                          " Still fit() is called.", RuntimeWarning)

    def fit_generator(self, *args,
                      disable_no_training_warning=False, **kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        if not disable_no_training_warning:
            # issue warning if not training is foreseen,
            # but is fit is still called.
            warnings.warn("This analyzer does not need to be trained."
                          " Still fit_generator() is called.", RuntimeWarning)

    def analyze(self, X):
        """
        Analyze the behavior of model on input `X`.

        :param X: Input as expected by model.
        """
        raise NotImplementedError()

    def _get_state(self):
        state = {
            "model_json": self._model.to_json(),
            "model_weights": self._model.get_weights(),
            "disable_model_checks": self._disable_model_checks,
        }
        return state

    def save(self):
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load`
        to resemble the analyzer.

        :return: The class name and the state.
        """
        state = self._get_state()
        class_name = self.__class__.__name__
        return class_name, state

    def save_npz(self, fname):
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load_npz`
        to resemble the analyzer.

        :param fname: The file's name.
        """
        class_name, state = self.save()
        np.savez(fname, **{"class_name": class_name,
                           "state": state})

    @classmethod
    def _state_to_kwargs(clazz, state):
        model_json = state.pop("model_json")
        model_weights = state.pop("model_weights")
        disable_model_checks = state.pop("disable_model_checks")
        assert len(state) == 0

        model = keras.models.model_from_json(model_json)
        model.set_weights(model_weights)
        return {"model": model,
                "disable_model_checks": disable_model_checks}

    @staticmethod
    def load(class_name, state):
        """
        Resembles an analyzer from the state created by
        :func:`analyzer.save()`.

        :param class_name: The analyzer's class name.
        :param state: The analyzer's state.
        """
        # Todo:do in a smarter way!
        import innvestigate.analyzer
        clazz = getattr(innvestigate.analyzer, class_name)

        kwargs = clazz._state_to_kwargs(state)
        return clazz(**kwargs)

    @staticmethod
    def load_npz(fname):
        """
        Resembles an analyzer from the file created by
        :func:`analyzer.save_npz()`.

        :param fname: The file's name.
        """
        f = np.load(fname)

        class_name = f["class_name"].item()
        state = f["state"].item()
        return AnalyzerBase.load(class_name, state)


###############################################################################
###############################################################################
###############################################################################


class TrainerMixin(object):
    """Mixin for analyzer that adapt to data.

    This convenience interface exposes a Keras like training routing
    to the user.
    """

    # todo: extend with Y
    def fit(self,
            X=None,
            batch_size=32,
            **kwargs):
        """
        Takes the same parameters as Keras's :func:`model.fit` function.
        """
        generator = iutils.BatchSequence(X, batch_size)
        return self._fit_generator(generator,
                                  **kwargs)

    def fit_generator(self, *args, **kwargs):
        """
        Takes the same parameters as Keras's :func:`model.fit_generator`
        function.
        """
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


class OneEpochTrainerMixin(TrainerMixin):
    """Exposes the same interface and functionality as :class:`TrainerMixin`
    except that the training is limited to one epoch.
    """

    def fit(self, *args, **kwargs):
        """
        Same interface as :func:`fit` of :class:`TrainerMixin` except that
        the parameter epoch is fixed to 1.
        """
        return super(OneEpochTrainerMixin, self).fit(*args, epochs=1, **kwargs)

    def fit_generator(self, *args, steps=None, **kwargs):
        """
        Same interface as :func:`fit_generator` of :class:`TrainerMixin` except that
        the parameter epoch is fixed to 1.
        """
        return super(OneEpochTrainerMixin, self).fit_generator(
            *args,
            steps_per_epoch=steps,
            epochs=1,
            **kwargs)


###############################################################################
###############################################################################
###############################################################################


class AnalyzerNetworkBase(AnalyzerBase):
    """Convenience interface for analyzers.

    This class provides helpful functionality to create analyzer's.
    Basically it:

    * takes the input model and adds a layer that selects
      the desired output neuron to analyze.
    * passes the new model to :func:`_create_analysis` which should
      return the analysis as Keras tensors.
    * compiles the function and serves the output to :func:`analyze` calls.
    * allows :func:`_create_analysis` to return tensors
      that are intercept for debugging purposes.

    :param neuron_selection_mode: How to select the neuron to analyze.
      Possible values are 'max_activation', 'index' for the neuron
      (expects indices at :func:`analyze` calls), 'all' take all neurons.
    """

    def __init__(self, model,
                 neuron_selection_mode="max_activation",
                 **kwargs):
        super(AnalyzerNetworkBase, self).__init__(model, **kwargs)

        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection parameter is not valid.")
        self._neuron_selection_mode = neuron_selection_mode

    def compile_analyzer(self):
        """
        Compiles the analyze functionality. If not called beforehand
        it will be called by :func:`analyze`.
        """
        model = self._model
        neuron_selection_mode = self._neuron_selection_mode

        neuron_selection_inputs = []
        model_inputs, model_output = model.inputs, model.outputs
        if len(model_output) > 1:
            raise ValueError("Only models with one output tensor are allowed.")

        # TODO Flatten the output before proceeding.
        if neuron_selection_mode == "max_activation":
            model_output = ilayers.Max()(model_output)
        elif neuron_selection_mode == "index":
            neuron_indexing = keras.layers.Input(shape=[None], dtype=np.int32)
            neuron_selection_inputs += [neuron_indexing]

            model_output = ilayers.Gather()(model_output+[neuron_indexing])
        elif neuron_selection_mode == "all":
            pass
        else:
            raise NotImplementedError()

        model = keras.models.Model(inputs=model_inputs+neuron_selection_inputs,
                                   outputs=model_output)
        tmp = self._create_analysis(model)
        if isinstance(tmp, tuple):
            if len(tmp) == 3:
                analysis_outputs, debug_outputs, constant_inputs = tmp
            elif len(tmp) == 2:
                analysis_outputs, debug_outputs = tmp
                constant_inputs = list()
            else:
                raise Exception("Unexpected output from _create_analysis.")
        else:
            analysis_outputs = iutils.to_list(tmp)
            constant_inputs, debug_outputs = list(), list()

        if neuron_selection_mode == "index":
            # Drop index, don't want to analyze that input.
            analysis_outputs = analysis_outputs[:-1]

        self._n_data_input = len(model_inputs)
        self._n_constant_input = len(constant_inputs)
        self._n_data_output = len(analysis_outputs)
        self._n_debug_output = len(debug_outputs)
        self._analyzer_model = keras.models.Model(
            inputs=model_inputs+neuron_selection_inputs+constant_inputs,
            outputs=analysis_outputs+debug_outputs)
        #self._analyzer_model.compile(optimizer="sgd", loss="mse")

    def _create_analysis(self, model):
        raise NotImplementedError()

    def _handle_debug_output(self, debug_values):
        raise NotImplementedError()

    def analyze(self, X, neuron_selection=None):
        """
        Same interface as :class:`Analyzer` besides

        :param neuron_selection: If neuron_selection_mode is 'index' this
          should be the indices for the chosen neuron(s).
        """
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
            ret = self._analyzer_model.predict_on_batch([X, neuron_selection])
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
    """Convenience class for analyzers that revert the model's structure.

    This class contains many helper functions around the graph
    reverse function :func:`innvestigate.utils.keras.graph.reverse_model`.

    The deriving classes should specify how the graph should be reverted
    by implementing the following functions:

    * :func:`_reverse_mapping(layer)` given a layer this function
      returns a reverse mapping for the layer as specified in
      :func:`innvestigate.utils.keras.graph.reverse_model` or None.
    * :func:`_default_reverse_mapping` defines the default
      reverse mapping.
    * :func:`_head_mapping` defines how the outputs of the model
      should be instantiated before the are passed to the reversed
      network.

    Furthermore other parameters of the function
    :func:`innvestigate.utils.keras.graph.reverse_model` can
    be changed by setting the according parameters of the
    init function:

    :param reverse_verbose: Be print information on the reverse process.
    :param reverse_clip_values: Clip the values that are passed along
      the reverted network. Expects tuple (min, max).
    :param reverse_project_bottleneck_layers: Project the value range
      of bottleneck tensors in the reverse network into another range.
    :param reverse_check_min_max_values: Print the min/max values
      observed in each tensor along the reverse network whenever
      :func:`analyze` is called.
    :param reverse_check_finite: Check if values passed along the
      reverse network are finite.
    :param reverse_reapply_on_copied_layers: See
      :func:`innvestigate.utils.keras.graph.reverse_model`.
    """


    # Should be specified by the base class.
    _conditional_mappings = []

    def __init__(self,
                 model,
                 reverse_verbose=False,
                 reverse_clip_values=False,
                 reverse_project_bottleneck_layers=False,
                 reverse_check_min_max_values=False,
                 reverse_check_finite=False,
                 reverse_reapply_on_copied_layers=False,
                 **kwargs):
        self._reverse_verbose = reverse_verbose
        self._reverse_clip_values = reverse_clip_values
        self._reverse_project_bottleneck_layers = (
            reverse_project_bottleneck_layers)
        self._reverse_check_min_max_values = reverse_check_min_max_values
        self._reverse_check_finite = reverse_check_finite
        self._reverse_reapply_on_copied_layers = (
            reverse_reapply_on_copied_layers)
        super(ReverseAnalyzerBase, self).__init__(model, **kwargs)

    def _reverse_mapping(self, layer):
        if isinstance(layer, (ilayers.Max, ilayers.Gather)):
            # Special layers added by AnalyzerNetworkBase
            # that should not be exposed to user.
            if isinstance(layer, ilayers.Max):
                return self._default_reverse_mapping
            if isinstance(layer, ilayers.Gather):
                # Gather second paramter is an index and has no gradient.
                def ignored_index_gradient(*args):
                    ret = self._default_reverse_mapping(*args,
                                                        mask=[True, False])
                    return iutils.to_list(ret)+[None]

                return ignored_index_gradient

        for condition, reverse_f in self._conditional_mappings:
            if condition(layer):
                return reverse_f
        return None

    def _default_reverse_mapping(self,
                                 Xs, Ys, reversed_Ys, reverse_state,
                                 mask=None):
        # The gradient.
        return ilayers.GradientWRT(len(Xs), mask=mask)(Xs+Ys+reversed_Ys)

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
            project_bottleneck_tensors=self._reverse_project_bottleneck_layers,
            return_all_reversed_tensors=return_all_reversed_tensors)

        if return_all_reversed_tensors:
            debug_tensors = []
            self._debug_tensors_indices = {}

            values = list(six.itervalues(ret[1]))
            mapping = {i: v["id"] for i, v in enumerate(values)}
            tensors = [v["final_tensor"] for v in values]
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
                tmp = iutils.to_list(ilayers.FiniteCheck()(tensors))
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
                  "((NodeID, TensorID), Value) - {}".format(tmp))

            indices = self._debug_tensors_indices["max"]
            tmp = debug_values[indices[0]:indices[1]]
            tmp = sorted([(self._reverse_tensors_mapping[i], v)
                          for i, v in enumerate(tmp)])
            print("Maximum values in tensors: "
                  "((NodeID, TensorID), Value) - {}".format(tmp))

        if self._reverse_check_finite:
            indices = self._debug_tensors_indices["finite"]
            tmp = debug_values[indices[0]:indices[1]]
            nfinite_tensors = np.flatnonzero(np.asarray(tmp) > 0)

            if len(nfinite_tensors) > 0:
                nfinite_tensors = sorted([self._reverse_tensors_mapping[i]
                                          for i in nfinite_tensors])
                print("Not finite values found in following nodes: "
                      "(NodeID, TensorID) - {}".format(nfinite_tensors))

    def _get_state(self):
        state = super(ReverseAnalyzerBase, self)._get_state()
        state.update({"reverse_verbose": self._reverse_verbose})
        state.update({"reverse_clip_values": self._reverse_clip_values})
        state.update({"reverse_project_bottleneck_layers":
                      self._reverse_project_bottleneck_layers})
        state.update({"reverse_check_min_max_values":
                      self._reverse_check_min_max_values})
        state.update({"reverse_check_finite": self._reverse_check_finite})
        state.update({"reverse_reapply_on_copied_layers":
                      self._reverse_reapply_on_copied_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        reverse_verbose = state.pop("reverse_verbose")
        reverse_clip_values = state.pop("reverse_clip_values")
        reverse_project_bottleneck_layers = (
            state.pop("reverse_project_bottleneck_layers"))
        reverse_check_min_max_values = (
            state.pop("reverse_check_min_max_values"))
        reverse_check_finite = state.pop("reverse_check_finite")
        reverse_reapply_on_copied_layers = (
            state.pop("reverse_reapply_on_copied_layers"))
        kwargs = super(ReverseAnalyzerBase, clazz)._state_to_kwargs(state)
        kwargs.update({"reverse_verbose": reverse_verbose,
                       "reverse_clip_values": reverse_clip_values,
                       "reverse_project_bottleneck_layers":
                       reverse_project_bottleneck_layers,
                       "reverse_check_min_max_values":
                       reverse_check_min_max_values,
                       "reverse_check_finite": reverse_check_finite,
                       "reverse_reapply_on_copied_layers":
                       reverse_reapply_on_copied_layers})
        return kwargs
