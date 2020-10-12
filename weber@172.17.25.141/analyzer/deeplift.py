# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import importlib
import numpy as np
import tempfile


from . import base
from .. import utils as iutils
from ..utils import keras as kutils


__all__ = [
    "DeepLIFTWrapper",
]


###############################################################################
###############################################################################
###############################################################################

#TODO: tf2.*
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
