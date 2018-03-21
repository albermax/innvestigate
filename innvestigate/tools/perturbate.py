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
import math
import warnings

import keras
from keras.utils import Sequence
from keras.utils.data_utils import OrderedEnqueuer, GeneratorEnqueuer


class Perturbation:
    """Perturbation of pixels based on analysis result."""

    def __init__(self, perturbation_function, ratio=0.05, region_shape=(9, 9), reduce_function=np.mean,
                 aggregation_function=np.max, pad_mode="reflect"):
        """
        :param perturbation_function: Defines the function with which the samples are perturbated. Can be a callable or a string that defines a predefined perturbation function.
        :type perturbation_function: callable or str
        :param ratio: Ratio of pixels to be perturbed.
        :type ratio: float
        :param reduce_function: Function to reduce the analysis result to one channel, e.g. mean or max function.
        :type reduce_function: callable
        """

        if isinstance(perturbation_function, str):
            if perturbation_function == "zeros":
                # This is equivalent to setting the perturbated values to the channel mean if the data are standardized.
                self.perturbation_function = np.zeros_like
            elif perturbation_function == "gaussian":
                self.perturbation_function = lambda x: x + np.random.normal(loc=0.0, scale=0.001,
                                                                            size=x.shape)  # TODO scale?
            elif perturbation_function == "mean":
                self.perturbation_function = np.mean
            else:
                raise ValueError("Perturbation function type '{}' not known.".format(perturbation_function))
        elif callable(perturbation_function):
            self.perturbation_function = perturbation_function
        else:
            raise TypeError("Cannot handle perturbation function of type {}.".format(type(perturbation_function)))

        self.ratio = ratio
        self.region_shape = region_shape
        self.reduce_function = reduce_function
        self.aggregation_function = aggregation_function

        self.pad_mode = pad_mode  # numpy.pad

    def compute_perturbation_mask(self, aggregated_regions, ratio):
        # Get indices and values
        thresholds = np.percentile(aggregated_regions, math.ceil(100 * (1 - ratio)), axis=(1, 2, 3), keepdims=True)
        perturbation_mask_regions = aggregated_regions >= thresholds

        return perturbation_mask_regions

    def expand_regions_to_pixels(self, regions):
        # Resize to pixels (repeat values).
        # (n, c, h_aggregated_region, w_aggregated_region) -> (n, c, h_aggregated_region, h_region, w_aggregated_region, w_region)
        regions_reshaped = np.expand_dims(np.expand_dims(regions, axis=3), axis=5)
        region_pixels = np.repeat(regions_reshaped, self.region_shape[0], axis=3)
        region_pixels = np.repeat(region_pixels, self.region_shape[1], axis=5)
        assert region_pixels.shape[0] == regions.shape[0] and region_pixels.shape[2:] == (
            regions.shape[2], self.region_shape[0], regions.shape[3], self.region_shape[1]), region_pixels.shape

        return region_pixels

    def reshape_region_pixels(self, region_pixels, target_shape):
        # Reshape to output shape
        pixels = region_pixels.reshape(target_shape)
        assert region_pixels.shape[0] == pixels.shape[0] and region_pixels.shape[1] == pixels.shape[1] and \
               region_pixels.shape[2] * region_pixels.shape[3] == pixels.shape[2] and region_pixels.shape[4] * \
               region_pixels.shape[5] == pixels.shape[3]
        return pixels

    def pad(self, analysis):
        pad_shape = self.region_shape - np.array(analysis.shape[2:]) % self.region_shape
        assert np.all(pad_shape < self.region_shape)

        # Pad half the window before and half after (on h and w axes)
        pad_shape_before = (pad_shape / 2).astype(int)
        pad_shape_after = pad_shape - pad_shape_before
        pad_shape = (
            (0, 0), (0, 0), (pad_shape_before[0], pad_shape_after[0]), (pad_shape_before[1], pad_shape_after[1]))
        analysis = np.pad(analysis, pad_shape, self.pad_mode)
        assert np.all(np.array(analysis.shape[2:]) % self.region_shape == 0), analysis.shape[2:]
        return analysis, pad_shape_before

    def reshape_to_regions(self, analysis):
        aggregated_shape = tuple((np.array(analysis.shape[2:]) / self.region_shape).astype(int))
        regions = analysis.reshape(
            (analysis.shape[0], analysis.shape[1], aggregated_shape[0], self.region_shape[0], aggregated_shape[1],
             self.region_shape[1]))
        return regions

    def aggregate_regions(self, analysis):
        regions = self.reshape_to_regions(analysis)
        aggregated_regions = self.aggregation_function(regions, axis=(3, 5))
        return aggregated_regions

    def perturbate_regions(self, x, perturbation_mask_regions, in_place=False):
        # Perturbate every region in tensor.
        # A single region (at region_x, region_y in sample) should be in mask[sample, channel, region_x, :, region_y, :]

        x_reshaped = self.reshape_to_regions(x)
        x_perturbated = x_reshaped if in_place else np.copy(x_reshaped)
        for sample_idx, channel_idx, region_row, region_col in np.ndindex(perturbation_mask_regions.shape):
            region = x_reshaped[sample_idx, channel_idx, region_row, :, region_col, :]
            region_mask = perturbation_mask_regions[sample_idx, channel_idx, region_row, region_col]
            if region_mask:
                x_perturbated[sample_idx, channel_idx, region_row, :, region_col, :] = self.perturbation_function(
                    region)
        x_perturbated = self.reshape_region_pixels(x_perturbated, x.shape)
        return x_perturbated

    def perturbate_on_batch(self, x, analysis):
        """
        :param x: Batch of images.
        :type x: numpy.ndarray
        :param analysis: Analysis of this batch.
        :type analysis: numpy.ndarray
        :param in_place: If true, samples are perturbated in place.
        :type in_place: bool
        :return: Batch of perturbated images
        :rtype: numpy.ndarray
        """
        assert analysis.shape == x.shape, analysis.shape
        original_shape = x.shape
        # reduce the analysis along channel axis -> n x 1 x h x w
        analysis = self.reduce_function(analysis, axis=1, keepdims=True)
        assert analysis.shape == (x.shape[0], 1, x.shape[2], x.shape[3]), analysis.shape

        padding = not np.all(np.array(analysis.shape[2:]) % self.region_shape == 0)
        if padding:
            analysis, pad_shape_before_analysis = self.pad(analysis)
            x, pad_shape_before_x = self.pad(x)
        aggregated_regions = self.aggregate_regions(analysis)

        # Compute perturbation mask (mask with ones where the input should be perturbated, zeros otherwise)
        perturbation_mask_regions = self.compute_perturbation_mask(aggregated_regions, self.ratio)
        # Perturbate each region
        x_perturbated = self.perturbate_regions(x, perturbation_mask_regions)

        # Crop the original image region to remove the padding
        if padding:
            x_perturbated = x_perturbated[:, :, pad_shape_before_x[0]:pad_shape_before_x[0] + original_shape[2],
                            pad_shape_before_x[1]:pad_shape_before_x[1] + original_shape[3]]

        return x_perturbated


class PerturbationAnalysis:
    """
    Performs the perturbation analysis.
    """

    def __init__(self, analyzer, model, generator, perturbation, preprocess, steps=1, recompute_analysis=True):
        """
        :param analyzer: Analyzer.
        :type analyzer: innvestigate.analyzer.base.AnalyzerBase
        :param model: Trained Keras model.
        :type model: keras.engine.training.Model
        :param generator: Data generator.
        :type generator: innvestigate.utils.BatchSequence
        :param perturbation: Instance of Perturbation class that performs the perturbation.
        :type perturbation: innvestigate.tools.Perturbation
        :param preprocess: Preprocessing function.
        :type preprocess: callable
        :param steps: Number of perturbation steps.
        :type steps: int
        :param recompute_analysis: If true, the analysis is recomputed after each perturbation step.
        :type recompute_analysis: bool
        """

        self.analyzer = analyzer
        self.model = model
        self.generator = generator
        self.perturbation = perturbation
        if not isinstance(perturbation, Perturbation):
            raise TypeError(type(perturbation))
        self.preprocess = preprocess
        self.steps = steps
        self.recompute_analysis = recompute_analysis

        if not self.recompute_analysis:
            raise NotImplementedError(
                "Not recomputing the analysis is not supported yet.")

    def evaluate_on_batch(self, x, y, sample_weight=None):
        """
        :param x: Samples.
        :type x: numpy.ndarray
        :param y: Labels.
        :type y: numpy.ndarray
        :param sample_weight: Sample weights.
        :type sample_weight: None
        :return: Test score.
        :rtype: list
        """
        if sample_weight is not None:
            raise NotImplementedError("Sample weighting is not supported yet.")  # TODO

        x = self.preprocess(x)
        a = self.analyzer.analyze(x)
        x_perturbated = self.perturbation.perturbate_on_batch(x, a)
        score = self.model.test_on_batch(x_perturbated, y)
        return score


    def evaluate_generator(self, generator, steps=None,
                           max_queue_size=10,
                           workers=1,
                           use_multiprocessing=False):
        """Evaluates the model on a data generator.

        The generator should return the same kind of data
        as accepted by `test_on_batch`.
        For documentation, refer to keras.engine.training.evaluate_generator (https://keras.io/models/model/)
        """

        steps_done = 0
        wait_time = 0.01
        all_outs = []
        batch_sizes = []
        is_sequence = isinstance(generator, Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
        if steps is None:
            if is_sequence:
                steps = len(generator)
            else:
                raise ValueError('`steps=None` is only valid for a generator'
                                 ' based on the `keras.utils.Sequence` class.'
                                 ' Please specify `steps` or use the'
                                 ' `keras.utils.Sequence` class.')
        enqueuer = None

        try:
            if workers > 0:
                if is_sequence:
                    enqueuer = OrderedEnqueuer(generator,
                                               use_multiprocessing=use_multiprocessing)
                else:
                    enqueuer = GeneratorEnqueuer(generator,
                                                 use_multiprocessing=use_multiprocessing,
                                                 wait_time=wait_time)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enqueuer.get()
            else:
                output_generator = generator

            while steps_done < steps:
                generator_output = next(output_generator)
                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))
                outs = self.evaluate_on_batch(x, y, sample_weight=sample_weight)

                if isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                if batch_size == 0:
                    raise ValueError('Received an empty batch. '
                                     'Batches should at least contain one item.')
                all_outs.append(outs)

                steps_done += 1
                batch_sizes.append(batch_size)

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        if not isinstance(outs, list):
            return np.average(np.asarray(all_outs),
                              weights=batch_sizes)
        else:
            averages = []
            for i in range(len(outs)):
                averages.append(np.average([out[i] for out in all_outs],
                                           weights=batch_sizes))
            return averages

    def compute_perturbation_analysis(self):
        scores = list()
        # Evaluate first on original data
        scores.append(self.model.evaluate_generator(self.generator))
        for step in range(self.steps):
            scores.append(self.evaluate_generator(self.generator))
        assert len(scores) == self.steps + 1
        return scores
