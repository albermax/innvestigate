# iNNvestigate neural networks!

![Different explanation methods on ImageNet.](https://github.com/albermax/innvestigate/raw/master/examples/images/analysis_grid.png)

## Note: The library is alpha-state and you might encounter issues using it. We are working on the next release. Please let us know if you find any bugs.

## Introduction

In the recent years neural networks furthered the state of the art in many domains like, e.g., object detection and speech recognition.
Despite the success neural networks are typically still treated as black boxes. Their internal workings are not fully understood and the basis for their predictions is unclear.
In the attempt to understand neural networks better several methods were proposed, e.g., Saliency, Deconvnet, GuidedBackprop, SmoothGrad, IntergratedGradients, LRP, PatternNet\&-Attribution.
Due to the lack of a reference implementations comparing them is a major effort.
This library addresses this by providing a common interface and out-of-the-box implementation for many analysis methods.
Our goal is to make analyzing neural networks' predictions easy!


**If you use this code please star the repository and cite the following paper:**
```
TODO: Add link to SW paper.
```

## Installation

iNNvestigate can be installed with the following commands.
The library is based on Keras and therefore requires a supported [Keras-backend](https://keras.io/backend/)
(Currently only Python 3.5, Tensorflow 1.8 and Cuda 9.x are supported.):

```bash
pip install git+https://github.com/albermax/innvestigate
# Installing Keras backend
pip install [tensorflow | theano | cntk]
```

To use the example scripts and notebooks one additionally needs to install the package matplotlib:

```bash
pip install matplotlib
```

The library's tests can be executed via:
```bash
git clone https://github.com/albermax/innvestigate.git
cd innvestigate
python setup.py test
```

The library was developed and tested on a Linux platform with Python 3.5, Tensorflow 1.8 and Cuda 9.x.

## Usage and Examples

The iNNvestigate library contains implementations for the following methods:

* *function:*
  * **gradient:** The gradient of the output neuron with respect to the input.
  * **smoothgrad:** [SmoothGrad](https://arxiv.org/abs/1706.03825)
  * **integrated_gradients:** [IntegratedGradients](https://arxiv.org/abs/1703.01365)
* *signal:*
  * **deconvnet:** [DeConvNet](https://arxiv.org/abs/1311.2901)
  * **guided:** [Guided BackProp](https://arxiv.org/abs/1412.6806)
  * **pattern.net:** [PatternNet](https://arxiv.org/abs/1705.05598)
* *interaction:*
  * **pattern.attribution:** [PatternAttribution](https://arxiv.org/abs/1705.05598)
  * **lrp.\*:** [LRP](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
* *miscellaneous:*
  * **input:** Returns the input.
  * **random:** Returns random Gaussian noise.

All the available methods have in common that they try to analyze the output of a specific neuron with respect to input to the neural network.
Typically one analyses the neuron with the largest activation in the output layer.
For example, given a Keras model, one can create a 'gradient' analyzer:

```python
import innvestigate

model = create_keras_model()

analyzer = innvestigate.create_analyzer("gradient", model)
```

and analyze the influence of the neural network's input on the output neuron by:

```python
analysis = analyzer.analyze(inputs)
```

To analyze a neuron with the index i, one can use the following scheme:

```python
analyzer = innvestigate.create_analyzer("gradient",
                                        model,
					neuron_selection_mode="index")
analysis = analyzer.analyze(inputs, i)
```

#### Trainable methods

Some methods like PatternNet and PatternAttribution are data-specific and need to be trained.
Given a data set with train and test data, this can be done in the following way:

```python
import innvestigate

analyzer = innvestigate.create_analyzer("pattern.net", model)
analyzer.fit(X_train)
analysis = analyzer.analyze(X_test)
```

#### Examples

In the directory [examples](https://github.com/albermax/innvestigate/blob/master/examples/) one can find different examples as Python scripts and as Jupyter notebooks:

* **[Imagenet](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/imagenet_example.ipynb):** shows how to use the different methods with VGG16 on ImageNet and how the reproduce the analysis grid above. This example uses pre-trained patterns for PatternNet.
* **[MNIST](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/mnist_example.ipynb):** shows how to train and use analyzers on MNIST.

## Contribution

If you would like to add your analysis method please get in touch with us!

## Releases

[Can be found here.](https://github.com/albermax/innvestigate/blob/master/VERSION.md)