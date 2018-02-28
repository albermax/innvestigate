# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import \
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import numpy as np
from heapq import nlargest


class Perturbation:
    def __init__(self, analyzer, perturbation_function, steps, recompute_analysis=False):
        self.analyzer = analyzer
        self.perturbation_function = perturbation_function
        self.steps = steps

        self.recompute_analysis = recompute_analysis

        # TODO
        if self.steps != 1:
            raise NotImplementedError("Only 1 perturbation step is supported.")  # TODO
        if self.recompute_analysis:
            raise NotImplementedError("Recomputation of analysis is not implemented yet.")  # TODO

    def calculate_thresholds_on_batch(self, a, nlargest_val, perturbate_largest=True):
        # TODO do not compute threshold but directly the indices (thresholds has advantages, though)
        def batch_flatten(x):
            # TODO this could go somewhere else
            # Flattens all but the first dimensions of a numpy array, i.e. flatten each sample in a batch for Keras tensors
            return x.reshape(x.shape[0], -1)

        # Sort the values and take the nlargest_val'th entry as threshold  TODO one could probably also directly operate on the indices
        thresholds = np.array([nlargest(nlargest_val, sample)[-1] for sample in batch_flatten(a)])
        return thresholds.reshape(a.shape[0], 1, 1, 1)

    def perturbate_on_batch(self, x, perturbation_mask):
        x_perturbated = np.copy(x)  # TODO optionally do it in place
        x_perturbated[perturbation_mask] = self.perturbation_function(x_perturbated[perturbation_mask])
        return x_perturbated

    def compute_on_batch(self, x, a):
        """
        :param x: Batch of images.
        :type x: numpy.ndarray
        :param a: Analysis of this batch.
        :type a: numpy.ndarray
        :return: Batch of perturbated images
        :rtype: numpy.ndarray
        """
        assert a.shape == x.shape, a.shape
        # reduce the analysis along channel axis -> n x 1 x h x w
        reduce = np.mean  # TODO possible alternatives could be K.max, should be a member variable or user defined
        a = reduce(a, axis=1, keepdims=True)
        assert a.shape == (x.shape[0], 1, x.shape[2], x.shape[3]), a.shape

        nlargest_val = int(0.05 * x.shape[2] * x.shape[3])  # TODO member variable
        thresholds = self.calculate_thresholds_on_batch(a,
                                                        nlargest_val)  # one threshold per sample. If a value is higher than the threshold, it should be perturbated
        assert thresholds.shape == (x.shape[0], 1, 1, 1), thresholds.shape

        perturbation_mask = a >= thresholds  # mask with ones where the input should be perturbated, zeros otherwise
        assert np.all(np.sum(perturbation_mask, axis=(1, 2,
                                                      3)) == nlargest_val), "Discrepancy between desired number of perturbations ({}) and actual number of perturbations ({}).".format(
            nlargest_val, np.sum(perturbation_mask, axis=(1, 2, 3)))

        x_perturbated = self.perturbate_on_batch(x, perturbation_mask)
        return x_perturbated
