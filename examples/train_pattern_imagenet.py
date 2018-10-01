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

import imp
import keras.backend
import keras.models
import keras.preprocessing.image
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import innvestigate
import innvestigate.tools
import innvestigate.utils as iutils
import innvestigate.utils.tests.networks.imagenet
import innvestigate.utils.visualizations as ivis


###############################################################################
###############################################################################
###############################################################################


base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))


# Path to train and validation images of Imagenet.
# Each directory should contain one directory for each class which contains
# the according images,
# see https://keras.io/preprocessing/image/#imagedatagenerator-class
# function flow_from_directory().
imagenet_train_dir = "/temp/datasets/imagenet/2012/train_set_small"
imagenet_val_dir = "/temp/datasets/imagenet/2012/train_set_small"


###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":

    netname = sys.argv[1] if len(sys.argv) > 1 else "vgg16"
    pattern_type = "relu"

    steps = None
    gpu_count = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    max_queue_size = 100
    workers = 4 * gpu_count
    use_multiprocessing = True
    print("GPU_COUNT", gpu_count)

    ###########################################################################
    # Build model.
    ###########################################################################
    tmp = getattr(innvestigate.applications.imagenet, netname)
    net = tmp(load_weights=True)
    model = keras.models.Model(inputs=net["in"], outputs=net["out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    modelp = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
    modelp.compile(optimizer="adam", loss="categorical_crossentropy",
                   metrics=["accuracy"])
    if gpu_count > 1:
        modelp = keras.utils.multi_gpu_model(modelp, gpus=gpu_count)
        modelp.compile(optimizer="adam",
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])

    ###########################################################################
    # Create data loaders.
    ###########################################################################

    if keras.backend.image_data_format() == "channels_first":
        target_size = net["input_shape"][2:4]
    else:
        target_size = net["input_shape"][1:3]

    def preprocess(X):
        X = X.copy()
        X = net["preprocess_f"](X)
        return X

    train_data_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess,
        horizontal_flip=True)
    test_data_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess)

    train_generator = train_data_generator.flow_from_directory(
        imagenet_train_dir,
        target_size=target_size,
        batch_size=32*gpu_count,
        class_mode=None)
    val_generator = test_data_generator.flow_from_directory(
        imagenet_val_dir,
        target_size=target_size,
        batch_size=32*gpu_count,
        class_mode='categorical')

    ###########################################################################
    # Evaluate and compute patterns.
    ###########################################################################

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
        pattern_type=pattern_type,
        compute_layers_in_parallel=True,
        gpus=gpu_count)
    patterns = pattern_computer.compute_generator(
        train_generator,
        steps_per_epoch=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=1)

    np.savez("%s_pattern_type_%s_tf_dim_ordering_tf_kernels.npz" %
             (netname, pattern_type),
             *patterns)

    ###########################################################################
    # Utility functions.
    ###########################################################################
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
    channels_first = keras.backend.image_data_format == "channels_first"

    def preprocess(X):
        X = X.copy()
        X = net["preprocess_f"](X)
        return X

    def postprocess(X):
        X = X.copy()
        X = iutils.postprocess_images(X,
                                      color_coding=color_conversion,
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
    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data()

    # Methods we use and some properties.
    methods = [
        # NAME             POSTPROCESSING     TITLE

        # Show input.
        ("input",                 {},                       image,   "Input"),

        # Function
        ("gradient",              {},                       graymap, "Gradient"),

        # Signal
        ("deconvnet",             {},                       bk_proj, "Deconvnet"),
        ("guided_backprop",       {},                       bk_proj, ("Guided", "Backprop"),),
        ("pattern.net",           {"patterns": patterns},   bk_proj, "PatterNet"),

        # Interaction
        ("pattern.attribution",   {"patterns": patterns},   heatmap, "PatternAttribution"),
        ("lrp.z",                 {},                       heatmap, "LRP-Z"),
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
            if not np.all(np.isfinite(a)):
                print("Image %i, analysis of %s not finite: nan %s inf %s" %
                      (i, methods[aidx][3],
                       np.any(np.isnan(a)), np.any(np.isinf(a))))
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

    file_name = "all_methods_%s_%s.pdf" % (netname, pattern_type)
    eutils.plot_image_grid(grid, row_labels, col_labels,
                           row_label_offset=50,
                           col_label_offset=-50,
                           usetex=True, file_name=file_name)
