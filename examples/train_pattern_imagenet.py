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


import matplotlib

matplotlib.use('Agg')

import imp
import keras.backend
import keras.models
import keras.preprocessing.image
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import os

import innvestigate
import innvestigate.tools
import innvestigate.utils.tests.networks.imagenet
import innvestigate.utils.visualizations as ivis


###############################################################################
###############################################################################
###############################################################################


# todo: make nicer!


def load_parameters(filename):
    f = np.load(filename)
    ret = [f["arr_%i" % i] for i in range(len(f.keys()))]
    return ret


def load_patterns(filename):
    f = np.load(filename)

    ret = {}
    for prefix in ["A", "r", "mu"]:
        l = sum([x.startswith(prefix) for x in f.keys()])
        ret.update({prefix: [f["%s_%i" % (prefix, i)] for i in range(l)]})

    return ret


def lasagne_weights_to_keras_weights(weights):
    ret = []
    for w in weights:
        if len(w.shape) < 4:
            ret.append(w)
        else:
            ret.append(w.transpose(2, 3, 1, 0))
    return ret


###############################################################################
###############################################################################
###############################################################################


base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))

param_file = "./imagenet_224_vgg_16.npz"
# Note those weights are CC 4.0:
# See http://www.robots.ox.ac.uk/~vgg/research/very_deep/
param_url = "https://www.dropbox.com/s/cvjj8x19hzya9oe/imagenet_224_vgg_16.npz?dl=1"


imagenet_train_dir = "/temp/datasets/imagenet/2012/train_set_small"
imagenet_val_dir = "/temp/datasets/imagenet/2012/train_set_small"


###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":
    # Download the necessary parameters for VGG16.
    eutils.download(param_url, param_file)

    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data()

    channels_first = keras.backend.image_data_format == "channels_first"

    steps = None
    gpu_count = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    max_queue_size = 100
    workers = 4 * gpu_count
    use_multiprocessing = True
    print("GPU_COUNT", gpu_count)

    def preprocess(X):
        X = X.copy()[None, :, :, :]
        X = ivis.preprocess_images(X, color_coding="RGBtoBGR")
        X = innvestigate.utils.tests.networks.imagenet.vgg16_custom_preprocess(X)
        return X[0]

    train_data_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess,
        horizontal_flip=True)
    test_data_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess)

    train_generator = train_data_generator.flow_from_directory(
        imagenet_train_dir,
        target_size=(224, 224),
        batch_size=32*gpu_count,
        class_mode=None)
    val_generator = test_data_generator.flow_from_directory(
        imagenet_val_dir,
        target_size=(224, 224),
        batch_size=32*gpu_count,
        class_mode='categorical')

    ###########################################################################
    # Build model.
    ###########################################################################
    parameters = lasagne_weights_to_keras_weights(load_parameters(param_file))
    vgg16 = innvestigate.utils.tests.networks.imagenet.vgg16_custom()
    print("Compile model1.")
    model = keras.models.Model(inputs=vgg16["in"], outputs=vgg16["out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.set_weights(parameters)
    print("Compile model2.")
    modelp = keras.models.Model(inputs=vgg16["in"], outputs=vgg16["sm_out"])
    modelp.compile(optimizer="adam",
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])
    modelp.set_weights(parameters)
    if gpu_count > 1:
        modelp = keras.utils.multi_gpu_model(modelp, gpus=gpu_count)
        modelp.compile(optimizer="adam",
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])

    # Check if all works correctly.
    print("Evaluate:")
    val_evaluation = modelp.evaluate_generator(
        val_generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)
    print(val_evaluation)

    print("Compute patterns:")
    pattern_computer = innvestigate.tools.PatternComputer(
        model,
        pattern_type="linear",
        compute_layers_in_parallel=True,
        gpus=gpu_count)
    patterns = pattern_computer.compute_generator(
        train_generator,
        steps_per_epoch=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=1)

    np.savez("trained_patterns.npz", *patterns)

    ###########################################################################
    # Utility functions.
    ###########################################################################

    def preprocess(X):
        X = X.copy()
        X = ivis.preprocess_images(X, color_coding="RGBtoBGR")
        X = innvestigate.utils.tests.networks.imagenet.vgg16_preprocess(X)
        return X

    def postprocess(X):
        X = X.copy()
        X = ivis.postprocess_images(X,
                                    color_coding="BGRtoRGB",
                                    channels_first=channels_first)
        return X

    def image(X):
        X = X.copy()
        return ivis.project(X, absmax=255.0, input_is_postive_only=True)

    def bk_proj(X):
        return ivis.project(X)

    def heatmap(X):
        return ivis.heatmap(X)

    def graymap(X):
        return ivis.graymap(np.abs(X), input_is_postive_only=True)

    ###########################################################################
    # Analysis.
    ###########################################################################

    # Methods we use and some properties.
    methods = [
        # NAME             POSTPROCESSING     TITLE

        # Show input.
        ("input",                 {},                       image,   "Input"),

        # Function
        ("gradient",              {},                       graymap, "Gradient"),
        ("smoothgrad",            {"noise_scale": 50},      graymap, "SmoothGrad"),
        ("integrated_gradients",  {},                       graymap, ("Integrated", "Gradients")),

        # Signal
        ("deconvnet",             {},                       bk_proj, "Deconvnet"),
        ("guided_backprop",       {},                       bk_proj, ("Guided", "Backprop"),),
        ("pattern.net",           {"patterns": patterns},   bk_proj, "PatterNet"),

        # Interaction
        ("pattern.attribution",   {"patterns": patterns},   heatmap, "PatternAttribution"),
        ("lrp.z_baseline",        {},                       heatmap, "LRP-Z"),
    ]

    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzers.append(innvestigate.create_analyzer(method[0],
                                                      model,
                                                      **method[1]))

    # Create analysis.
    analysis = np.zeros([len(images), len(analyzers), 224, 224, 3])
    text = []
    for i, (image, y) in enumerate(images):
        image = image[None, :, :, :]
        # Predict label.
        x = preprocess(image)
        prob = modelp.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        text.append((r"\textbf{%s}" % label_to_class_name[y],
                     r"\textit{(%.2f)}" % prob.max(),
                     r"\textit{%s}" % label_to_class_name[y_hat]))

        for aidx, analyzer in enumerate(analyzers):
            is_input_analyzer = methods[aidx][0] == "input"
            # Analyze.
            a = analyzer.analyze(image if is_input_analyzer else x)
            # Postprocess.
            if not is_input_analyzer:
                a = postprocess(a)
            a = methods[aidx][2](a)
            analysis[i, aidx] = a[0]

    ###########################################################################
    # Plot the analysis.
    ###########################################################################

    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    row_labels = text
    col_labels = [method[3] for method in methods]

    eutils.plot_image_grid(grid, row_labels, col_labels,
                           row_label_offset=50,
                           col_label_offset=-50,
                           usetex=True, file_name="all_methods.pdf")
