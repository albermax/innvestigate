# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import range


###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.models
import tensorflow.keras.engine.topology


from ... import utils as iutils


__all__ = [
    "TestAnalysisHelper",
]


###############################################################################
###############################################################################
###############################################################################


class TestAnalysisHelper(object):

    def __init__(self, model, analyzer, weights=None):
        """ Helper class for retrieving output and analysis in test cases.
        

        :param model: A Keras layer object or a list of layer objects.
          In this case a sequntial model will be build. The first layer
          must have set input_shape or batch_input_shape.
          Alternatively a tuple with input and output tensors, in which
          case the tensorflow.keras modle api will be used.
        :param analyzer: Either an analyzer class or a function
          that takes a tensorflow.keras model and returns an analyzer.
        :param weights: After creating the model set the given weights.
        """

        if isinstance(model, tensorflow.keras.engine.topology.Layer):
            model = [model]

        if isinstance(model, list):
            self._model = tensorflow.keras.models.Sequential(model)
        else:
            self._model = tensorflow.keras.models.Model(*model)

        self._input_shapes = iutils.to_list(self._model.input_shape)

        if weights is not None:
            self._model.set_weights(weights)

        self._analyzer = analyzer(self._model)

    @property
    def weights(self):
        return self._model.get_weights()
    
    def run(self, inputs):
        """Runs the model given the inputs.

        :return: Tuple with model output and analyzer output.
        """
        return_list = True
        if not isinstance(inputs, list):
            return_list = False
            inputs = iutils.to_list(inputs)

        augmented = []
        for i in range(len(inputs)):
            if len(inputs[i].shape) == len(self._input_shapes[i])-1:
                # Augment by batch axis.
                augmented.append(i)
                inputs[i] = inputs[i].reshape((1,)+inputs[i].shape)

        outputs = iutils.to_list(self._model.predict_on_batch(inputs))
        analysis = iutils.to_list(self._analyzer.analyze(inputs))

        for i in augmented:
            # Remove batch axis.
            outputs[i] = outputs[i][0]
            analysis[i] = analysis[i][0]
        
        if return_list:
            return outputs, analysis
        else:
            return outputs[0], analysis[0]
