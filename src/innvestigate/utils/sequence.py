from __future__ import annotations

import math
from typing import Callable

import tensorflow.keras.utils as kutils

from innvestigate.backend import to_list
from innvestigate.backend.types import OptionalList, Tensor

__all__ = [
    "BatchSequence",
    "TargetAugmentedSequence",
]


class BatchSequence(kutils.Sequence):
    """Batch sequence generator.

    Take a (list of) input tensors and a batch size
    and creates a generators that creates a sequence of batches.

    :param Xs: One or a list of tensors. First axis needs to have same length.
    :param batch_size: Batch size. Default 32.
    """

    def __init__(self, Xs: OptionalList[Tensor], batch_size: int = 32) -> None:
        self.Xs: list[Tensor] = to_list(Xs)
        self.single_tensor: bool = len(Xs) == 1
        self.batch_size: int = batch_size

        if not self.single_tensor:
            for X in self.Xs[1:]:
                assert X.shape[0] == self.Xs[0].shape[0]
        super().__init__()

    def __len__(self) -> int:
        return int(math.ceil(float(len(self.Xs[0])) / self.batch_size))

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor]:
        ret: list[Tensor] = [
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
        self, sequence: list[Tensor], augment_f: Callable[[list[Tensor]], list[Tensor]]
    ) -> None:
        self.sequence = sequence
        self.augment_f = augment_f

        super().__init__()

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, idx: int) -> tuple[list[Tensor], list[Tensor]]:
        inputs = self.sequence[idx]
        if isinstance(inputs, tuple):  # TODO: check if this can be removed
            assert len(inputs) == 1
            inputs = inputs[0]

        targets = self.augment_f(to_list(inputs))
        return inputs, targets
