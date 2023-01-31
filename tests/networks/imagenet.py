"""Load networks from Keras applications for dryrunning.
By default, weights are just randomly initialized.
"""
from __future__ import annotations

import tensorflow.keras.applications as kapps

from innvestigate.backend.graph import model_wo_softmax
from innvestigate.backend.types import Model

__all__ = [
    "vgg16",
    "vgg19",
    "resnet50",
    "inception_v3",
    "inception_resnet_v2",
    "densenet121",
    "densenet169",
    "densenet201",
    "nasnet_large",
    "nasnet_mobile",
]

WEIGHTS = None  # either None (random initialization) or "imagenet"
ACTIVATION = None  # leave as None to return the logits of the output layer


def vgg16() -> Model:
    return kapps.VGG16(
        weights=WEIGHTS,
        # classifier_activation=ACTIVATION,
    )


def vgg19() -> Model:
    return kapps.VGG19(
        weights=WEIGHTS,
        # classifier_activation=ACTIVATION,
    )


def resnet50() -> Model:
    model = kapps.ResNet50(
        weights=WEIGHTS,
        # classifier_activation=ACTIVATION,
    )
    return model_wo_softmax(model)


###############################################################################


def inception_v3() -> Model:
    return kapps.InceptionV3(
        weights=WEIGHTS,
        # classifier_activation=ACTIVATION,
    )


def inception_resnet_v2() -> Model:
    return kapps.InceptionResNetV2(
        weights=WEIGHTS,
        # classifier_activation=ACTIVATION,
    )


###############################################################################


def densenet121() -> Model:
    model = kapps.DenseNet121(
        weights=WEIGHTS,
    )
    return model_wo_softmax(model)


def densenet169() -> Model:
    model = kapps.DenseNet169(
        weights=WEIGHTS,
    )
    return model_wo_softmax(model)


def densenet201() -> Model:
    model = kapps.DenseNet201(
        weights=WEIGHTS,
    )
    return model_wo_softmax(model)


###############################################################################


def nasnet_large() -> Model:
    model = kapps.NASNetLarge(
        weights=WEIGHTS,
    )
    return model_wo_softmax(model)


def nasnet_mobile() -> Model:
    model = kapps.NASNetMobile(
        weights=WEIGHTS,
    )
    return model_wo_softmax(model)
