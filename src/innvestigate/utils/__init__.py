from __future__ import annotations

import tensorflow.keras.backend as kbackend

from innvestigate.backend.types import Tensor

__all__ = [
    "preprocess_images",
    "postprocess_images",
]


def preprocess_images(images: Tensor, color_coding: str = None) -> Tensor:
    """Image preprocessing

    Takes a batch of images and:
    * Adjust the color axis to the Keras format.
    * Fixes the color coding.

    :param images: Batch of images with 4 axes.
    :param color_coding: Determines the color coding.
      Can be None, 'RGBtoBGR' or 'BGRtoRGB'.
    :return: The preprocessed batch.
    """

    ret: Tensor = images
    image_data_format: str = kbackend.image_data_format()

    # TODO: not very general:
    channels_first: bool = images.shape[1] in [1, 3]
    if image_data_format == "channels_first" and not channels_first:
        ret = ret.transpose(0, 3, 1, 2)
    if image_data_format == "channels_last" and channels_first:
        ret = ret.transpose(0, 2, 3, 1)

    assert color_coding in [None, "RGBtoBGR", "BGRtoRGB"]
    if color_coding in ["RGBtoBGR", "BGRtoRGB"]:
        if image_data_format == "channels_first":
            ret = ret[:, ::-1, :, :]
        if image_data_format == "channels_last":
            ret = ret[:, :, :, ::-1]

    return ret


def postprocess_images(
    images: Tensor, color_coding: str = None, channels_first: bool = None
) -> Tensor:
    """Image postprocessing

    Takes a batch of images and reverts the preprocessing.

    :param images: A batch of images with 4 axes.
    :param color_coding: The initial color coding,
      see :func:`preprocess_images`.
    :param channels_first: The output channel format.
    :return: The postprocessed images.
    """

    ret: Tensor = images
    image_data_format: str = kbackend.image_data_format()

    assert color_coding in [None, "RGBtoBGR", "BGRtoRGB"]
    if color_coding in ["RGBtoBGR", "BGRtoRGB"]:
        if image_data_format == "channels_first":
            ret = ret[:, ::-1, :, :]
        if image_data_format == "channels_last":
            ret = ret[:, :, :, ::-1]

    if image_data_format == "channels_first" and not channels_first:
        ret = ret.transpose(0, 2, 3, 1)
    if image_data_format == "channels_last" and channels_first:
        ret = ret.transpose(0, 3, 1, 2)

    return ret
