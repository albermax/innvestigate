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
import heapq

from innvestigate.utils.visualizations import batch_flatten

class Perturbation:
    def __init__(self, analyzer, perturbation_function, ratio=0.05, steps=1, reduce_function=np.mean,
                 recompute_analysis=False):
        self.analyzer = analyzer
        self.perturbation_function = perturbation_function
        self.ratio = ratio  # How many of the pixels should be perturbated
        self.steps = steps
        self.reduce_function = reduce_function
        self.recompute_analysis = recompute_analysis

        if self.steps != 1:
            raise NotImplementedError("Only 1 perturbation step is supported.")  # TODO
        if self.recompute_analysis:
            raise NotImplementedError("Recomputation of analysis is not implemented yet.")  # TODO

    def calculate_thresholds_on_batch(self, a, num_perturbated_pixels):
        # TODO do not compute threshold but directly the indices (thresholds has advantages, though)
        # Sort the values and take the num_perturbated_pixels'th entry as threshol
        thresholds = np.array([heapq.nlargest(num_perturbated_pixels, sample)[-1] for sample in batch_flatten(a)])
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
        a = self.reduce_function(a, axis=1, keepdims=True)
        assert a.shape == (x.shape[0], 1, x.shape[2], x.shape[3]), a.shape

        num_perturbated_pixels = int(self.ratio * x.shape[2] * x.shape[3])
        # Calculate one threshold per sample. If a value is higher than the threshold, it should be perturbated.
        thresholds = self.calculate_thresholds_on_batch(a, num_perturbated_pixels)
        assert thresholds.shape == (x.shape[0], 1, 1, 1), thresholds.shape

        perturbation_mask = a >= thresholds  # mask with ones where the input should be perturbated, zeros otherwise
        try:
            assert np.all(np.sum(perturbation_mask, axis=(1, 2,
                                                          3)) == num_perturbated_pixels), "Discrepancy between desired number of perturbations ({}) and actual number of perturbations ({}).".format(
                num_perturbated_pixels, np.sum(perturbation_mask, axis=(1, 2, 3)))
        except AssertionError as error:
            pass  # TODO
        x_perturbated = self.perturbate_on_batch(x, perturbation_mask)
        return x_perturbated
