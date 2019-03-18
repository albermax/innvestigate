# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import importlib
import keras.backend as K
import keras.layers
import numpy as np
import tempfile
import warnings


from . import base
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph


__all__ = [
    "DeepLIFT",
    "DeepLIFTWrapper",
]


###############################################################################
###############################################################################
###############################################################################


def _create_deeplift_rules(reference_mapping, approximate_gradient=True):
    def RescaleRule(Xs, Ys, As, reverse_state, local_references={}):
        if approximate_gradient:
            def rescale_f(x):
                a, dx, dy, g = x
                return K.switch(K.less(K.abs(dx), K.epsilon()), g, a*(dy/dx))
        else:
            def rescale_f(x):
                a, dx, dy, _ = x
                return a*(dy/(dx + K.epsilon()))

        grad = ilayers.GradientWRT(len(Xs))
        rescale = keras.layers.Lambda(rescale_f)

        Xs_references = [
            reference_mapping.get(x, local_references.get(x, None))
            for x in Xs
        ]
        Ys_references = [
            reference_mapping.get(x, local_references.get(x, None))
            for x in Ys
        ]

        Xs_differences = [keras.layers.Subtract()([x, r])
                          for x, r in zip(Xs, Xs_references)]
        Ys_differences = [keras.layers.Subtract()([x, r])
                          for x, r in zip(Ys, Ys_references)]
        gradients = iutils.to_list(grad(Xs+Ys+As))

        return [rescale([a, dx, dy, g])
                for a, dx, dy, g
                in zip(As, Xs_differences, Ys_differences, gradients)]

    def LinearRule(Xs, Ys, As, reverse_state):
        if approximate_gradient:
            def switch_f(x):
                dx, a, g = x
                return K.switch(K.less(K.abs(dx), K.epsilon()), g, a)
        else:
            def switch_f(x):
                _, a, _ = x
                return a

        grad = ilayers.GradientWRT(len(Xs))
        switch = keras.layers.Lambda(switch_f)

        Xs_references = [reference_mapping[x] for x in Xs]

        Ys_references = [reference_mapping[x] for x in Ys]

        Xs_differences = [keras.layers.Subtract()([x, r])
                          for x, r in zip(Xs, Xs_references)]
        Ys_differences = [keras.layers.Subtract()([x, r])
                          for x, r in zip(Ys, Ys_references)]

        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(As, Ys_differences)]

        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Ys+tmp))

        # Re-weight relevance with the input values.
        tmp = [keras.layers.Multiply()([a, b])
               for a, b in zip(Xs_differences, tmp)]

        # only the gradient
        gradients = iutils.to_list(grad(Xs+Ys+As))

        return [switch([dx, a, g])
                for dx, a, g
                in zip(Xs_differences, tmp, gradients)]

    return RescaleRule, LinearRule


class DeepLIFT(base.ReverseAnalyzerBase):
    """DeepLIFT-rescale algorithm

    This class implements the DeepLIFT algorithm using
    the rescale rule (as in DeepExplain (Ancona et.al.)).

    WARNING: This implementation contains bugs.

    :param model: A Keras model.
    """

    def __init__(self, model, *args, **kwargs):
        warnings.warn("This implementation contains bugs.")
        self._reference_inputs = kwargs.pop("reference_inputs", 0)
        self._approximate_gradient = kwargs.pop(
            "approximate_gradient", True)
        self._add_model_softmax_check()
        super(DeepLIFT, self).__init__(model, *args, **kwargs)

    def _prepare_model(self, model):
        ret = super(DeepLIFT, self)._prepare_model(model)
        # Store analysis input to create reference inputs.
        self._analysis_inputs = ret[1]
        return ret

    def _create_reference_activations(self, model):
        self._model_execution_trace = kgraph.trace_model_execution(model)
        layers, execution_list, outputs = self._model_execution_trace

        self._reference_activations = {}

        # Create references and graph inputs.
        tmp = kutils.broadcast_np_tensors_to_keras_tensors(
            model.inputs, self._reference_inputs)
        tmp = [K.variable(x) for x in tmp]

        constant_reference_inputs = [
            keras.layers.Input(tensor=x, shape=K.int_shape(x)[1:])
            for x in tmp
        ]

        for k, v in zip(model.inputs, constant_reference_inputs):
            self._reference_activations[k] = v

        for k, v in zip(self._analysis_inputs, self._analysis_inputs):
            self._reference_activations[k] = v

        # Compute intermediate states.
        for layer, Xs, Ys in execution_list:
            activations = [self._reference_activations[x] for x in Xs]

            if isinstance(layer, keras.layers.InputLayer):
                # Special case. Do nothing.
                next_activations = activations
            else:
                next_activations = iutils.to_list(
                    kutils.apply(layer, activations))

            assert len(next_activations) == len(Ys)
            for k, v in zip(Ys, next_activations):
                self._reference_activations[k] = v

        return constant_reference_inputs

    def _create_analysis(self, model, *args, **kwargs):
        constant_reference_inputs = self._create_reference_activations(model)

        RescaleRule, LinearRule = _create_deeplift_rules(
            self._reference_activations, self._approximate_gradient)

        # Kernel layers.
        self._add_conditional_reverse_mapping(
            lambda l: kchecks.contains_kernel(l),
            LinearRule,
            name="deeplift_kernel_layer",
        )

        # Activation layers
        self._add_conditional_reverse_mapping(
            lambda l: (not kchecks.contains_kernel(l) and
                       kchecks.contains_activation(l)),
            RescaleRule,
            name="deeplift_activation_layer",
        )

        tmp = super(DeepLIFT, self)._create_analysis(
            model, *args, **kwargs)

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

        return (analysis_outputs,
                debug_outputs,
                constant_inputs+constant_reference_inputs)

    def _head_mapping(self, X):
        return keras.layers.Subtract()([X, self._reference_activations[X]])

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
            return_all_reversed_tensors=return_all_reversed_tensors,
            execution_trace=self._model_execution_trace)

    def _get_state(self):
        state = super(DeepLIFT, self)._get_state()
        state.update({"reference_inputs": self._reference_inputs})
        state.update({"approximate_gradient": self._approximate_gradient})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        reference_inputs = state.pop("reference_inputs")
        approximate_gradient = state.pop("approximate_gradient")
        kwargs = super(DeepLIFT, clazz)._state_to_kwargs(state)
        kwargs.update({"reference_inputs": reference_inputs})
        kwargs.update({"approximate_gradient": approximate_gradient})
        return kwargs


###############################################################################
###############################################################################
###############################################################################


class DeepLIFTWrapper(base.AnalyzerNetworkBase):
    """Wrapper around DeepLIFT package

    This class wraps the DeepLIFT package.
    For further explanation of the parameters check out:
    https://github.com/kundajelab/deeplift

    :param model: A Keras model.
    :param nonlinear_mode: The nonlinear mode parameter.
    :param reference_inputs: The reference input used for DeepLIFT.
    :param verbose: Verbosity of the DeepLIFT package.

    :note: Requires the deeplift package.
    """

    def __init__(self, model, **kwargs):
        self._nonlinear_mode = kwargs.pop("nonlinear_mode", "rescale")
        self._reference_inputs = kwargs.pop("reference_inputs", 0)
        self._verbose = kwargs.pop("verbose", False)
        self._add_model_softmax_check()

        try:
            self._deeplift_module = importlib.import_module("deeplift")
        except ImportError:
            raise ImportError("To use DeepLIFTWrapper please install "
                              "the python module 'deeplift', e.g.: "
                              "'pip install deeplift'")

        super(DeepLIFTWrapper, self).__init__(model, **kwargs)

    def _create_deep_lift_func(self):
        # Store model and load into deeplift format.
        kc = importlib.import_module("deeplift.conversion.kerasapi_conversion")
        modes = self._deeplift_module.layers.NonlinearMxtsMode

        key = self._nonlinear_mode
        nonlinear_mxts_mode = {
            "genomics_default": modes.DeepLIFT_GenomicsDefault,
            "reveal_cancel": modes.RevealCancel,
            "rescale": modes.Rescale,
        }[key]

        with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
            self._model.save(f.name)
            deeplift_model = kc.convert_model_from_saved_files(
                f.name, nonlinear_mxts_mode=nonlinear_mxts_mode,
                verbose=self._verbose)

        # Create function with respect to input layers
        def fix_name(s):
            return s.replace(":", "_")

        score_layer_names = [fix_name(l.name) for l in self._model.inputs]
        if len(self._model.outputs) > 1:
            raise ValueError("Only a single output layer is supported.")
        tmp = self._model.outputs[0]._keras_history
        target_layer_name = fix_name(tmp[0].name+"_%i" % tmp[1])
        self._func = deeplift_model.get_target_contribs_func(
            find_scores_layer_name=score_layer_names,
            pre_activation_target_layer_name=target_layer_name)
        self._references = kutils.broadcast_np_tensors_to_keras_tensors(
            self._model.inputs, self._reference_inputs)

    def _analyze_with_deeplift(self, X, neuron_idx, batch_size):
        return self._func(task_idx=neuron_idx,
                          input_data_list=X,
                          batch_size=batch_size,
                          input_references_list=self._references,
                          progress_update=1000000)

    def analyze(self, X, neuron_selection=None):
        if not hasattr(self, "_deep_lift_func"):
            self._create_deep_lift_func()

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
            if neuron_selection.size != 1:
                # The code allows to select multiple neurons.
                raise ValueError("One neuron can be selected with DeepLIFT.")

            neuron_idx = neuron_selection[0]
            analysis = self._analyze_with_deeplift(X, neuron_idx, len(X[0]))

            # Parse the output.
            ret = []
            for x, analysis_for_x in zip(X, analysis):
                tmp = np.stack([a for a in analysis_for_x])
                tmp = tmp.reshape(x.shape)
                ret.append(tmp)
        elif self._neuron_selection_mode == "max_activation":
            neuron_idx = np.argmax(self._model.predict_on_batch(X), axis=1)

            analysis = []
            # run for each batch with its respective max activated neuron
            for i, ni in enumerate(neuron_idx):
                # slice input tensors
                tmp = [x[i:i+1] for x in X]
                analysis.append(self._analyze_with_deeplift(tmp, ni, 1))

            # Parse the output.
            ret = []
            for i, x in enumerate(X):
                tmp = np.stack([a[i] for a in analysis]).reshape(x.shape)
                ret.append(tmp)
        else:
            raise ValueError("Only neuron_selection_mode index or "
                             "max_activation are supported.")

        if isinstance(ret, list) and len(ret) == 1:
            ret = ret[0]
        return ret

    def _get_state(self):
        state = super(DeepLIFTWrapper, self)._get_state()
        state.update({"nonlinear_mode": self._nonlinear_mode})
        state.update({"reference_inputs": self._reference_inputs})
        state.update({"verbose": self._verbose})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        nonlinear_mode = state.pop("nonlinear_mode")
        reference_inputs = state.pop("reference_inputs")
        verbose = state.pop("verbose")
        kwargs = super(DeepLIFTWrapper, clazz)._state_to_kwargs(state)
        kwargs.update({
            "nonlinear_mode": nonlinear_mode,
            "reference_inputs": reference_inputs,
            "verbose": verbose,
        })
        return kwargs
