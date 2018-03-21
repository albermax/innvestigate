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
                self.perturbation_function = lambda x: x + np.random.normal(loc=0.0, scale=1.0,
                                                                            size=x.shape)  # TODO scale?
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

    def reshape_regions_to_pixels(self, regions, target_shape):
        # Resize to pixels (repeat values). (n, c, hr, wr) -> (n, c,
        regions_reshaped = np.expand_dims(np.expand_dims(regions, axis=3), axis=5)
        pixels = np.repeat(regions_reshaped, self.region_shape[0], axis=3)
        pixels = np.repeat(pixels, self.region_shape[1], axis=5)
        pixels = pixels.reshape(target_shape)
        assert regions.shape[0] == pixels.shape[0] and regions.shape[1] == pixels.shape[1]
        return pixels

    def perturbate_on_batch(self, x, analysis, in_place=True):
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
        # reduce the analysis along channel axis -> n x 1 x h x w
        analysis = self.reduce_function(analysis, axis=1, keepdims=True)
        assert analysis.shape == (x.shape[0], 1, x.shape[2], x.shape[3]), analysis.shape

        # Padding
        if not np.all(np.array(analysis.shape[2:]) % self.region_shape == 0):
            pad_shape = self.region_shape - np.array(analysis.shape[2:]) % self.region_shape
            assert np.all(pad_shape < self.region_shape)

            # Pad half the window before and half after (on h and w axes)
            pad_shape_before = (pad_shape / 2).astype(int)
            pad_shape_after = pad_shape - pad_shape_before
            pad_shape = (
                (0, 0), (0, 0), (pad_shape_before[0], pad_shape_after[0]), (pad_shape_before[1], pad_shape_after[1]))
            analysis = np.pad(analysis, pad_shape, self.pad_mode)
            assert np.all(np.array(analysis.shape[2:]) % self.region_shape == 0), analysis.shape[2:]
        aggregated_shape = tuple((np.array(analysis.shape[2:]) / self.region_shape).astype(int))
        regions = analysis.reshape(
            (analysis.shape[0], analysis.shape[1], aggregated_shape[0], self.region_shape[0], aggregated_shape[1],
             self.region_shape[1]))
        aggregated_regions = self.aggregation_function(regions, axis=(3, 5))

        # Compute perturbation mask (mask with ones where the input should be perturbated, zeros otherwise)
        perturbation_mask_regions = self.compute_perturbation_mask(aggregated_regions, self.ratio)
        perturbation_mask = self.reshape_regions_to_pixels(perturbation_mask_regions, analysis.shape)
        assert np.all(np.sum(perturbation_mask, axis=(1, 2, 3)) / (perturbation_mask.shape[2] * perturbation_mask.shape[
            3]) >= self.ratio), "Too little perturbed pixels ({}).".format(np.sum(perturbation_mask, axis=(1, 2, 3)))
        # Crop the original image region to remove the padding
        perturbation_mask = perturbation_mask[:x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]]

        # Perturbate
        x_perturbated = x if in_place else np.copy(x)
        x_perturbated[perturbation_mask] = self.perturbation_function(x_perturbated[perturbation_mask])

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
