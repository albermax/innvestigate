"""Load pretrained patterns for keras.applications models."""

from __future__ import annotations

import warnings
from builtins import range

import numpy as np
import tensorflow.keras.utils as kutils

__all__ = [
    "load_patterns",
]


PATTERNS = {
    "vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/15lip81fzvbgkaa/vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",  # noqa
        "hash": "8c2abe648e116a93fd5027fab49177b0",
    },
    "vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/nc5empj78rfe9hm/vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",  # noqa
        "hash": "3258b6c64537156afe75ca7b3be44742",
    },
}


def _get_patterns_info(netname: str, pattern_type: str):
    if pattern_type is True:
        pattern_type = "relu"

    file_name = f"{netname}_pattern_type_{pattern_type}_tf_dim_ordering_tf_kernels.npz"

    return {
        "file_name": file_name,
        "url": PATTERNS[file_name]["url"],
        "hash": PATTERNS[file_name]["hash"],
    }


def load_patterns(netname: str, pattern_type: str):
    try:
        pattern_info = _get_patterns_info(netname, pattern_type)
    except KeyError:
        warnings.warn("There are no patterns for network '%s'." % netname)

    patterns_path = kutils.get_file(
        pattern_info["file_name"],
        pattern_info["url"],
        cache_subdir="innvestigate_patterns",
        hash_algorithm="md5",
        file_hash=pattern_info["hash"],
    )
    patterns_file = np.load(patterns_path)
    patterns = [patterns_file["arr_%i" % i] for i in range(len(patterns_file.keys()))]

    return patterns
