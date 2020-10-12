"""Example applications for image classifcation.

Each function returns a pretrained MNIST model.
The models are based on https://doi.org/10.1371/journal.pone.0130140
and http://jmlr.org/papers/v17/15-618.html.

"""
# TODO: rename in, sm_out, out to input_tensors, output_tensors,
# TODO: softmax_output_tenors
# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import os
import numpy as np

import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models
from tensorflow.keras.models import load_model, clone_model
#from tensorflow.keras.utils import get_file


__all__ = [
    "pretrained_plos_long_relu",
    "pretrained_plos_short_relu",
    "pretrained_plos_long_tanh",
    "pretrained_plos_short_tanh",
]


###############################################################################
###############################################################################
###############################################################################

# pre-trained models from [https://doi.org/10.1371/journal.pone.0130140 , http://jmlr.org/papers/v17/15-618.html]
PRETRAINED_MODELS = {"pretrained_plos_long_relu":
                        {"file":"plos-mnist-rect-long.h5",
                         "url" : "https://www.dropbox.com/s/26w7i58qqcuosn4/plos-mnist-rect-long.h5"
                        },
                     "pretrained_plos_short_relu":
                        {"file":"plos-mnist-rect-short.h5",
                         "url":"https://www.dropbox.com/s/89nvwyls55xycmw/plos-mnist-rect-short.h5"
                        },
                     "pretrained_plos_long_tanh":
                        {"file":"plos-mnist-tanh-long.h5",
                         "url":"https://www.dropbox.com/s/61e3a4gdbjo9bca/plos-mnist-tanh-long.h5"
                        },
                     "pretrained_plos_short_tanh":
                        {"file":"plos-mnist-tanh-short.h5",
                         "url":"https://www.dropbox.com/s/foqv60kot0retfr/plos-mnist-tanh-short.h5"
                        },
                    }


def _load_pretrained_net(modelname, new_input_shape):
    filename = PRETRAINED_MODELS[modelname]["file"]
    urlname = PRETRAINED_MODELS[modelname]["url"]
    #model_path = get_file(fname=filename, origin=urlname) #TODO: FIX! corrupts the file?
    model_path = os.path.expanduser('~') + "/.keras/models/" + filename


    #workaround the more elegant, but dysfunctional solution.
    if not os.path.isfile(model_path):
        model_dir = os.path.dirname(model_path)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        os.system("wget {} &&  mv -v {} {}".format(urlname, filename, model_path))


    model = load_model(model_path)
    #create replacement input layer with new shape.
    model.layers[0] = keras_layers.InputLayer(input_shape=new_input_shape, name="input_1")
    for l in model.layers:
        l.name = "%s_workaround" % l.name
    model = keras_models.Sequential(layers=model.layers)

    model_w_sm = clone_model(model)

    #NOTE: perform forward pass to fix a keras 2.2.0 related issue with improper weight initialization
    #See: https://github.com/albermax/innvestigate/issues/88
    x_dummy = np.zeros(new_input_shape)[None, ...]
    model_w_sm.predict(x_dummy)

    model_w_sm.set_weights(model.get_weights())
    model_w_sm.add(keras_layers.Activation("softmax"))
    return model, model_w_sm


def pretrained_plos_long_relu(input_shape, **kwargs):
    return _load_pretrained_net("pretrained_plos_long_relu", input_shape)

def pretrained_plos_short_relu(input_shape, **kwargs):
    return _load_pretrained_net("pretrained_plos_short_relu", input_shape)

def pretrained_plos_long_tanh(input_shape, **kwargs):
    return _load_pretrained_net("pretrained_plos_long_tanh", input_shape)

def pretrained_plos_short_tanh(input_shape, **kwargs):
    return _load_pretrained_net("pretrained_plos_short_tanh", input_shape)
