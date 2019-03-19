# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import zip
import six

###############################################################################
###############################################################################
###############################################################################


import keras.backend as K
import keras.layers
import keras.models
import numpy as np
import warnings


from .. import layers as ilayers
from .. import utils as iutils
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph


__all__ = [
    "NotAnalyzeableModelException",
    "AnalyzerBase",

    "TrainerMixin",
    "OneEpochTrainerMixin",

    "AnalyzerNetworkBase",
    "ReverseAnalyzerBase"
]


###############################################################################
###############################################################################
###############################################################################


class NotAnalyzeableModelException(Exception):
    """Indicates that the model cannot be analyzed by an analyzer."""
    pass


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

    def __init__(self, model, disable_model_checks=False):
        self._model = model
        self._disable_model_checks = disable_model_checks

        self._do_model_checks()

    def _add_model_check(self, check, message, check_type="exception"):
        if getattr(self, "_model_check_done", False):
            raise Exception("Cannot add model check anymore."
                            " Check was already performed.")

        if not hasattr(self, "_model_checks"):
            self._model_checks = []

        check_instance = {
            "check": check,
            "message": message,
            "type": check_type,
        }
        self._model_checks.append(check_instance)

    def _do_model_checks(self):
        model_checks = getattr(self, "_model_checks", [])

        if not self._disable_model_checks and len(model_checks) > 0:
            check = [x["check"] for x in model_checks]
            types = [x["type"] for x in model_checks]
            messages = [x["message"] for x in model_checks]

            checked = kgraph.model_contains(self._model, check)
            tmp = zip(iutils.to_list(checked), messages, types)

            for checked_layers, message, check_type in tmp:
                if len(checked_layers) > 0:
                    tmp_message = ("%s\nCheck triggerd by layers: %s" %
                                   (message, checked_layers))

                    if check_type == "exception":
                        raise NotAnalyzeableModelException(tmp_message)
                    elif check_type == "warning":
                        # TODO(albermax) only the first warning will be shown
                        warnings.warn(tmp_message)
                    else:
                        raise NotImplementedError()

        self._model_check_done = True

    def fit(self, *args, **kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        disable_no_training_warning = kwargs.pop("disable_no_training_warning",
                                                 False)
        if not disable_no_training_warning:
            # issue warning if not training is foreseen,
            # but is fit is still called.
            warnings.warn("This analyzer does not need to be trained."
                          " Still fit() is called.", RuntimeWarning)

    def fit_generator(self, *args, **kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
          called despite no training is needed.
        """
        disable_no_training_warning = kwargs.pop("disable_no_training_warning",
                                                 False)
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

    def fit_generator(self, *args, **kwargs):
        """
        Same interface as :func:`fit_generator` of :class:`TrainerMixin` except that
        the parameter epoch is fixed to 1.
        """
        steps = kwargs.pop("steps", None)
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
    :param allow_lambda_layers: Allow the model to contain lambda layers.
    """

    def __init__(self, model,
                 neuron_selection_mode="max_activation",
                 allow_lambda_layers=False,
                 **kwargs):
        if neuron_selection_mode not in ["max_activation", "index", "all"]:
            raise ValueError("neuron_selection parameter is not valid.")
        self._neuron_selection_mode = neuron_selection_mode

        self._allow_lambda_layers = allow_lambda_layers
        self._add_model_check(
            lambda layer: (not self._allow_lambda_layers and
                           isinstance(layer, keras.layers.core.Lambda)),
            ("Lamda layers are not allowed. "
             "To force use set allow_lambda_layers parameter."),
            check_type="exception",
        )

        self._special_helper_layers = []

        super(AnalyzerNetworkBase, self).__init__(model, **kwargs)

    def _add_model_softmax_check(self):
        """
        Adds check that prevents models from containing a softmax.
        """
        self._add_model_check(
            lambda layer: kchecks.contains_activation(
                layer, activation="softmax"),
            "This analysis method does not support softmax layers.",
            check_type="exception",
        )

    def _prepare_model(self, model):
        """
        Prepares the model to analyze before it gets actually analyzed.

        This class adds the code to select a specific output neuron.
        """
        neuron_selection_mode = self._neuron_selection_mode
        model_inputs = model.inputs

        model_output = model.outputs
        if len(model_output) > 1:
            raise ValueError("Only models with one output tensor are allowed.")
        analysis_inputs = []
        stop_analysis_at_tensors = []

        # Flatten to form (batch_size, other_dimensions):
        if K.ndim(model_output[0]) > 2:
            model_output = keras.layers.Flatten()(model_output)

        if neuron_selection_mode == "max_activation":
            l = ilayers.Max(name="iNNvestigate_max")
            model_output = l(model_output)
            self._special_helper_layers.append(l)
        elif neuron_selection_mode == "index":
            neuron_indexing = keras.layers.Input(
                batch_shape=[None, None], dtype=np.int32,
                name='iNNvestigate_neuron_indexing')
            self._special_helper_layers.append(
                neuron_indexing._keras_history[0])
            analysis_inputs.append(neuron_indexing)
            # The indexing tensor should not be analyzed.
            stop_analysis_at_tensors.append(neuron_indexing)

            l = ilayers.GatherND(name="iNNvestigate_gather_nd")
            model_output = l(model_output+[neuron_indexing])
            self._special_helper_layers.append(l)
        elif neuron_selection_mode == "all":
            pass
        else:
            raise NotImplementedError()

        model = keras.models.Model(inputs=model_inputs+analysis_inputs,
                                   outputs=model_output)
        return model, analysis_inputs, stop_analysis_at_tensors

    def create_analyzer_model(self):
        """
        Creates the analyze functionality. If not called beforehand
        it will be called by :func:`analyze`.
        """
        model_inputs = self._model.inputs
        tmp = self._prepare_model(self._model)
        model, analysis_inputs, stop_analysis_at_tensors = tmp
        self._analysis_inputs = analysis_inputs
        self._prepared_model = model

        tmp = self._create_analysis(
            model, stop_analysis_at_tensors=stop_analysis_at_tensors)
        if isinstance(tmp, tuple):
            if len(tmp) == 3:
                analysis_outputs, debug_outputs, constant_inputs = tmp
            elif len(tmp) == 2:
                analysis_outputs, debug_outputs = tmp
                constant_inputs = list()
            elif len(tmp) == 1:
                analysis_outputs = iutils.to_list(tmp[0])
                constant_inputs, debug_outputs = list(), list()
            else:
                raise Exception("Unexpected output from _create_analysis.")
        else:
            analysis_outputs = tmp
            constant_inputs, debug_outputs = list(), list()

        analysis_outputs = iutils.to_list(analysis_outputs)
        debug_outputs = iutils.to_list(debug_outputs)
        constant_inputs = iutils.to_list(constant_inputs)

        self._n_data_input = len(model_inputs)
        self._n_constant_input = len(constant_inputs)
        self._n_data_output = len(analysis_outputs)
        self._n_debug_output = len(debug_outputs)
        self._analyzer_model = keras.models.Model(
            inputs=model_inputs+analysis_inputs+constant_inputs,
            outputs=analysis_outputs+debug_outputs)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        """
        Interface that needs to be implemented by a derived class.

        This function is expected to create a Keras graph that creates
        a custom analysis for the model inputs given the model outputs.

        :param model: Target of analysis.
        :param stop_analysis_at_tensors: A list of tensors where to stop the
          analysis. Similar to stop_gradient arguments when computing the
          gradient of a graph.
        :return: Either one-, two- or three-tuple of lists of tensors.
          * The first list of tensors represents the analysis for each
            model input tensor. Tensors present in stop_analysis_at_tensors
            should be omitted.
          * The second list, if present, is a list of debug tensors that will
            be passed to :func:`_handle_debug_output` after the analysis
            is executed.
          * The third list, if present, is a list of constant input tensors
            added to the analysis model.
        """
        raise NotImplementedError()

    def _handle_debug_output(self, debug_values):
        raise NotImplementedError()

    def analyze(self, X, neuron_selection=None):
        """
        Same interface as :class:`Analyzer` besides

        :param neuron_selection: If neuron_selection_mode is 'index' this
          should be an integer with the index for the chosen neuron.
        """
        if not hasattr(self, "_analyzer_model"):
            self.create_analyzer_model()

        X = iutils.to_list(X)

        if(neuron_selection is not None and
           self._neuron_selection_mode != "index"):
            raise ValueError("Only neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")
        if(neuron_selection is None and
           self._neuron_selection_mode == "index"):
            raise ValueError("neuron_selection_mode 'index' expects "
                             "the neuron_selection parameter.")

        if self._neuron_selection_mode == "index":
            neuron_selection = np.asarray(neuron_selection).flatten()
            if neuron_selection.size == 1:
                neuron_selection = np.repeat(neuron_selection, len(X[0]))

            # Add first axis indices for gather_nd
            neuron_selection = np.hstack(
                (np.arange(len(neuron_selection)).reshape((-1, 1)),
                 neuron_selection.reshape((-1, 1)))
            )
            ret = self._analyzer_model.predict_on_batch(X+[neuron_selection])
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
        state.update({"allow_lambda_layers": self._allow_lambda_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        neuron_selection_mode = state.pop("neuron_selection_mode")
        allow_lambda_layers = state.pop("allow_lambda_layers")
        kwargs = super(AnalyzerNetworkBase, clazz)._state_to_kwargs(state)
        kwargs.update({
            "neuron_selection_mode": neuron_selection_mode,
            "allow_lambda_layers": allow_lambda_layers
        })
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
    :func:`innvestigate.utils.keras.graph.reverse_model` can
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
      :func:`innvestigate.utils.keras.graph.reverse_model`.
    """

    def __init__(self,
                 model,
                 reverse_verbose=False,
                 reverse_clip_values=False,
                 reverse_project_bottleneck_layers=False,
                 reverse_check_min_max_values=False,
                 reverse_check_finite=False,
                 reverse_keep_tensors=False,
                 reverse_reapply_on_copied_layers=False,
                 **kwargs):
        self._reverse_verbose = reverse_verbose
        self._reverse_clip_values = reverse_clip_values
        self._reverse_project_bottleneck_layers = (
            reverse_project_bottleneck_layers)
        self._reverse_check_min_max_values = reverse_check_min_max_values
        self._reverse_check_finite = reverse_check_finite
        self._reverse_keep_tensors = reverse_keep_tensors
        self._reverse_reapply_on_copied_layers = (
            reverse_reapply_on_copied_layers)
        super(ReverseAnalyzerBase, self).__init__(model, **kwargs)

    def _gradient_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        mask = [x not in reverse_state["stop_mapping_at_tensors"] for x in Xs]
        return ilayers.GradientWRT(len(Xs), mask=mask)(Xs+Ys+reversed_Ys)

    def _reverse_mapping(self, layer):
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
            self, condition, mapping, priority=-1, name=None):
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
        if getattr(self, "_reverse_mapping_applied", False):
            raise Exception("Cannot add conditional mapping "
                            "after first application.")

        if not hasattr(self, "_conditional_reverse_mappings"):
            self._conditional_reverse_mappings = {}

        if priority not in self._conditional_reverse_mappings:
            self._conditional_reverse_mappings[priority] = []

        tmp = {"condition": condition, "mapping": mapping, "name": name}
        self._conditional_reverse_mappings[priority].append(tmp)

    def _apply_conditional_reverse_mappings(self, layer):
        mappings = getattr(self, "_conditional_reverse_mappings", {})
        self._reverse_mapping_applied = True

        # Search for mapping. First consider ones with highest priority,
        # inside priority in order of adding.
        sorted_keys = sorted(mappings.keys())[::-1]
        for key in sorted_keys:
            for mapping in mappings[key]:
                if mapping["condition"](layer):
                    return mapping["mapping"]

        return None

    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        """
        Fallback function to map reversed_Ys to reversed_Xs
        (which should contain tensors of the same shape and type).
        """
        return self._gradient_reverse_mapping(
            Xs, Ys, reversed_Ys, reverse_state)

    def _head_mapping(self, X):
        """
        Map output tensors to new values before passing
        them into the reverted network.
        """
        return X

    def _postprocess_analysis(self, X):
        return X

    def _reverse_model(self,
                       model,
                       stop_analysis_at_tensors=[],
                       return_all_reversed_tensors=False):
        return kgraph.reverse_model(
            model,
            reverse_mappings=self._reverse_mapping,
            default_reverse_mapping=self._default_reverse_mapping,
            head_mapping=self._head_mapping,
            stop_mapping_at_tensors=stop_analysis_at_tensors,
            verbose=self._reverse_verbose,
            clip_all_reversed_tensors=self._reverse_clip_values,
            project_bottleneck_tensors=self._reverse_project_bottleneck_layers,
            return_all_reversed_tensors=return_all_reversed_tensors)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        return_all_reversed_tensors = (
            self._reverse_check_min_max_values or
            self._reverse_check_finite or
            self._reverse_keep_tensors
        )
        ret = self._reverse_model(
            model,
            stop_analysis_at_tensors=stop_analysis_at_tensors,
            return_all_reversed_tensors=return_all_reversed_tensors)

        if return_all_reversed_tensors:
            ret = (self._postprocess_analysis(ret[0]), ret[1])
        else:
            ret = self._postprocess_analysis(ret)

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

            if self._reverse_keep_tensors:
                self._debug_tensors_indices["keep"] = (
                    len(debug_tensors),
                    len(debug_tensors)+len(tensors))
                debug_tensors += tensors

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

        if self._reverse_keep_tensors:
            indices = self._debug_tensors_indices["keep"]
            tmp = debug_values[indices[0]:indices[1]]
            tmp = sorted([(self._reverse_tensors_mapping[i], v)
                          for i, v in enumerate(tmp)])
            self._reversed_tensors = tmp

    def _get_state(self):
        state = super(ReverseAnalyzerBase, self)._get_state()
        state.update({"reverse_verbose": self._reverse_verbose})
        state.update({"reverse_clip_values": self._reverse_clip_values})
        state.update({"reverse_project_bottleneck_layers":
                      self._reverse_project_bottleneck_layers})
        state.update({"reverse_check_min_max_values":
                      self._reverse_check_min_max_values})
        state.update({"reverse_check_finite": self._reverse_check_finite})
        state.update({"reverse_keep_tensors": self._reverse_keep_tensors})
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
        reverse_keep_tensors = state.pop("reverse_keep_tensors")
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
                       "reverse_keep_tensors": reverse_keep_tensors,
                       "reverse_reapply_on_copied_layers":
                       reverse_reapply_on_copied_layers})
        return kwargs
