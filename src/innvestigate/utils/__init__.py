from __future__ import annotations

import math
from typing import Callable, List, Tuple, TypeVar, Union

import tensorflow.keras.backend as kbackend
import tensorflow.keras.utils as kutils

from innvestigate.utils.types import OptionalList, Tensor

__all__ = [
    "to_list",
    "unpack_singleton",
    "apply_mask",
    "BatchSequence",
    "TargetAugmentedSequence",
    "preprocess_images",
    "postprocess_images",
]


T = TypeVar("T")  # Generic type, can be anything


def to_list(X: OptionalList[T]) -> List[T]:
    """Wraps tensor `X` into a list, if it isn't a list of Tensors yet."""
    if isinstance(X, list):
        return X
    return [X]


def unpack_singleton(x: OptionalList[T]) -> OptionalList[T]:
    """Gets the first element of a list if it has only one value.

    Otherwise return the list.

    # Argument
        x: A list or singleton.

    # Returns
        The same list or the first element.
    """
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x


def apply_mask(Xs: List[T], mask: List[bool]) -> List[T]:
    """Apply mask to list `Xs`, keeping only the elements for which
    mask is True.

    :param Xs: List to be masked.
    :type Xs: List[T]
    :param mask: Mask applied to `Xs`. If True, corresponding entry in `Xs` is kept.
    :type mask: List[bool]
    :return: Masked list.
    :rtype: List[T]
    """
    assert len(Xs) == len(mask)
    return [x for x, keep in zip(Xs, mask) if keep]


###############################################################################


class BatchSequence(kutils.Sequence):
    """Batch sequence generator.

    Take a (list of) input tensors and a batch size
    and creates a generators that creates a sequence of batches.

    :param Xs: One or a list of tensors. First axis needs to have same length.
    :param batch_size: Batch size. Default 32.
    """

    def __init__(self, Xs: OptionalList[Tensor], batch_size: int = 32) -> None:
        self.Xs: List[Tensor] = to_list(Xs)
        self.single_tensor: bool = len(Xs) == 1
        self.batch_size: int = batch_size

        if not self.single_tensor:
            for X in self.Xs[1:]:
                assert X.shape[0] == self.Xs[0].shape[0]
        super().__init__()

    def __len__(self) -> int:
        return int(math.ceil(float(len(self.Xs[0])) / self.batch_size))

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor]]:
        ret: List[Tensor] = [
            X[idx * self.batch_size : (idx + 1) * self.batch_size] for X in self.Xs
        ]

        if self.single_tensor:
            return ret[0]
        return tuple(ret)


class TargetAugmentedSequence(kutils.Sequence):
    """Augments a sequence with a target on the fly.

    Takes a sequence/generator and a function that
    creates on the fly for each batch a target.
    The generator takes a batch from that sequence,
    computes the target and returns both.

    :param sequence: A sequence or generator.
    :param augment_f: Takes a batch and returns a target.
    """

    def __init__(
        self, sequence: List[Tensor], augment_f: Callable[[List[Tensor]], List[Tensor]]
    ) -> None:
        self.sequence = sequence
        self.augment_f = augment_f

        super().__init__()

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], List[Tensor]]:
        inputs = self.sequence[idx]
        if isinstance(inputs, tuple):  # TODO: check if this can be removed
            assert len(inputs) == 1
            inputs = inputs[0]

        targets = self.augment_f(to_list(inputs))
        return inputs, targets


###############################################################################


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
