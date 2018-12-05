# [iNNvestigate neural networks!](https://github.com/albermax/innvestigate) [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=iNNvestigate%20neural%20networks!&url=https://github.com/albermax/innvestigate&hashtags=iNNvestigate,artificialintelligence,machinelearning,deeplearning,datascience)


[![GitHub package version](https://img.shields.io/badge/Version-v1.0.6-green.svg)](https://github.com/albermax/innvestigate)
[![Keras package version](https://img.shields.io/badge/KerasVersion-v2.2.4-green.svg)](https://github.com/albermax/innvestigate)
[![License: BSD-2](https://img.shields.io/badge/License-BSD--2-blue.svg)](https://github.com/albermax/innvestigate/blob/master/LICENSE)

![Different explanation methods on ImageNet.](https://github.com/albermax/innvestigate/raw/master/examples/images/analysis_grid.png)

## Table of contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Usage and Examples](#usage-and-examples)
* [More documentation](#more-documentation)
* [Contributing](#contributing)
* [Releases](#releases)

## Introduction

In the recent years neural networks furthered the state of the art in many domains like, e.g., object detection and speech recognition.
Despite the success neural networks are typically still treated as black boxes. Their internal workings are not fully understood and the basis for their predictions is unclear.
In the attempt to understand neural networks better several methods were proposed, e.g., Saliency, Deconvnet, GuidedBackprop, SmoothGrad, IntergratedGradients, LRP, PatternNet\&-Attribution.
Due to the lack of a reference implementations comparing them is a major effort.
This library addresses this by providing a common interface and out-of-the-box implementation for many analysis methods.
Our goal is to make analyzing neural networks' predictions easy!


### If you use this code please star the repository and cite the following paper:

**["iNNvestigate neural networks!"](http://arxiv.org/abs/1808.04260)([http://arxiv.org/abs/1808.04260](http://arxiv.org/abs/1808.04260)) by Maximilian Alber, Sebastian Lapuschkin, Philipp Seegerer, Miriam H&auml;gele, Kristof T. Sch&uuml;tt, Gr&eacute;goire Montavon, Wojciech Samek, Klaus-Robert M&uuml;ller, Sven D&auml;hne, Pieter-Jan Kindermans**

## Installation

iNNvestigate can be installed with the following commands.
The library is based on Keras and therefore requires a supported [Keras-backend](https://keras.io/backend/)
o(Currently only the Tensorflow backend is supported. We test with Python 3.6, Tensorflow 1.12 and Cuda 9.x.):

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
python setup.py test -m precommit
```

## Usage and Examples

The iNNvestigate library contains implementations for the following methods:

* *function:*
  * **gradient:** The gradient of the output neuron with respect to the input.
  * **smoothgrad:** [SmoothGrad](https://arxiv.org/abs/1706.03825)
* *signal:*
  * **deconvnet:** [DeConvNet](https://arxiv.org/abs/1311.2901)
  * **guided:** [Guided BackProp](https://arxiv.org/abs/1412.6806)
  * **pattern.net:** [PatternNet](https://arxiv.org/abs/1705.05598)
* *attribution:*
  * **pattern.attribution:** [PatternAttribution](https://arxiv.org/abs/1705.05598)
  * **occlusion:** (Coming soon.)
  * **input_t_gradient:** Input \* Gradient
  * **integrated_gradients:** [IntegratedGradients](https://arxiv.org/abs/1703.01365)
  * **deep_taylor[.bounded]:** [DeepTaylor](https://www.sciencedirect.com/science/article/pii/S0031320316303582?via%3Dihub)
  * **lrp.\*:** [LRP](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
  * **deeplift:** [DeepLIFT (Beta) (as propsoed by Ancona et. al.)](https://openreview.net/forum?id=Sy21R9JAW)
  * **deeplift.wrapper:** [DeepLIFT (wrapper around original code, slower)](http://proceedings.mlr.press/v70/shrikumar17a.html)
* *miscellaneous:*
  * **input:** Returns the input.
  * **random:** Returns random Gaussian noise.

**The intention behind iNNvestigate is to make it easy to use analysis methods, but it is not to explain the underlying concepts and assumptions. Please, read the according publication(s) when using a certain method and when publishing please cite the according paper(s) (as well as the [iNNvestigate paper](https://arxiv.org/abs/1808.04260)). Thank you!**

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

### Tutorials

In the directory [examples](https://github.com/albermax/innvestigate/blob/master/examples/) one can find different examples as Python scripts and as Jupyter notebooks:

* **[Introduction to iNNvestigate](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/introduction.ipynb):** shows how to use **iNNvestigate**.
* **[Comparing methods on MNIST](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/mnist_compare_methods.ipynb):** shows how to train and compare analyzers on MNIST.
* **[Comparing output neurons on MNIST](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/mnist_neuron_selection.ipynb):** shows how to analyze the prediction of different classes on MNIST.
* **[Comparing methods on ImageNet](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/imagenet_compare_methods.ipynb):** shows how to compare analyzers on ImageNet.
* **[Comparing networks on ImageNet](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/imagenet_compare_networks.ipynb):** shows how to compare analyzes for different networks on ImageNet.
* **[Sentiment Analysis](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/sentiment_analysis.ipynb)**.
* **[Development with iNNvestigate](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/introduction_development.ipynb):** shows how to develop with **iNNvestigate**.
* **[Perturbation Analysis](https://github.com/albermax/innvestigate/blob/master/examples/notebooks/mnist_perturbation.ipynb)**.
---

**To use ImageNet examples one must download example pictures first ([script](https://github.com/albermax/innvestigate/blob/master/examples/images/wget_imagenet_2011_samples.sh)).**

## More documentation

... can be found here: [https://innvestigate.readthedocs.io/en/latest/](https://innvestigate.readthedocs.io/en/latest/)

## Contributing

If you would like to contribute or add your analysis method please get in touch with us!

## Releases

[Can be found here.](https://github.com/albermax/innvestigate/blob/master/VERSION.md)
