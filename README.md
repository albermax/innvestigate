<div align="center">
  <a href="https://github.com/albermax/innvestigate/">
    <img
      src="docs/assets/logo.svg"
      alt="iNNvestigate Logo"
      height="100"
    />
  </a>
  <br />
  <p>
    <h1>
      <b>
        iNNvestigate neural networks!
      </b>
    </h1>

[![Documentation](https://img.shields.io/badge/Documentation-stable-blue.svg)](https://innvestigate.readthedocs.io/en/latest/)
[![Build Status](https://github.com/albermax/innvestigate/actions/workflows/ci.yml/badge.svg)](https://github.com/albermax/innvestigate/actions/workflows/ci.yml)
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=iNNvestigate%20neural%20networks!&url=https://github.com/albermax/innvestigate&hashtags=iNNvestigate,artificialintelligence,machinelearning,deeplearning,datascience)

[![PyPI package version](https://img.shields.io/pypi/v/innvestigate)](https://pypi.org/project/innvestigate/)
[![GitHub package version](https://img.shields.io/github/v/tag/albermax/innvestigate)](https://github.com/albermax/innvestigate/tags)
[![License: BSD-2](https://img.shields.io/badge/License-BSD--2-purple.svg)](https://github.com/albermax/innvestigate/blob/master/LICENSE)
[![Black](https://img.shields.io/badge/code_style-black-black.svg)](https://github.com/psf/black)

[![Python](https://img.shields.io/pypi/pyversions/innvestigate.svg)](https://badge.fury.io/py/innvestigate)
[![TensorFlow package version](https://img.shields.io/badge/TensorFlow-2.6_--_2.14-orange.svg)](https://github.com/albermax/innvestigate)

![Different explanation methods on ImageNet.](https://github.com/albermax/innvestigate/raw/master/examples/images/analysis_grid.png)

  </p>
</div>

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
In the attempt to understand neural networks better several methods were proposed, e.g., Saliency, Deconvnet, GuidedBackprop, SmoothGrad, IntegratedGradients, LRP, PatternNet and PatternAttribution.
Due to the lack of a reference implementations comparing them is a major effort.
This library addresses this by providing a common interface and out-of-the-box implementation for many analysis methods.
Our goal is to make analyzing neural networks' predictions easy!


### If you use this code please star the repository and cite the following paper:

[Alber, M., Lapuschkin, S., Seegerer, P., Hägele, M., Schütt, K. T., Montavon, G., Samek, W., Müller, K. R., Dähne, S., & Kindermans, P. J. (2019). **iNNvestigate neural networks!** Journal of Machine Learning Research, 20.](https://jmlr.org/papers/v20/18-540.html)
  ```
  @article{JMLR:v20:18-540,
  author  = {Maximilian Alber and Sebastian Lapuschkin and Philipp Seegerer and Miriam H{{\"a}}gele and Kristof T. Sch{{\"u}}tt and Gr{{\'e}}goire Montavon and Wojciech Samek and Klaus-Robert M{{\"u}}ller and Sven D{{\"a}}hne and Pieter-Jan Kindermans},
  title   = {iNNvestigate Neural Networks!},
  journal = {Journal of Machine Learning Research},
  year    = {2019},
  volume  = {20},
  number  = {93},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v20/18-540.html}
  }
  ```

## Installation

iNNvestigate is based on Keras and TensorFlow 2 and can be installed with the following commands:

```bash
pip install innvestigate
```

**Please note that iNNvestigate currently requires disabling TF2's eager execution.**

To use the example scripts and notebooks one additionally needs to install the package matplotlib:

```bash
pip install matplotlib
```

The library's tests can be executed via `pytest`. The easiest way to do reproducible development on iNNvestigate is to install all dev dependencies via [Poetry](https://python-poetry.org):
```bash
git clone https://github.com/albermax/innvestigate.git
cd innvestigate

poetry install
poetry run pytest
```

## Usage and Examples

The iNNvestigate library contains implementations for the following methods:

* *function:*
  * **gradient:** The gradient of the output neuron with respect to the input.
  * **smoothgrad:** [SmoothGrad](https://arxiv.org/abs/1706.03825) averages the gradient over number of inputs with added noise.
* *signal:*
  * **deconvnet:** [DeConvNet](https://arxiv.org/abs/1311.2901) applies a ReLU in the gradient computation instead of the gradient of a ReLU.
  * **guided:** [Guided BackProp](https://arxiv.org/abs/1412.6806) applies a ReLU in the gradient computation additionally to the gradient of a ReLU.
  * **pattern.net:** [PatternNet](https://arxiv.org/abs/1705.05598) estimates the input signal of the output neuron. (*Note: not available in iNNvestigate 2.0*)
* *attribution:*
  * **input_t_gradient:** Input \* Gradient
  * **deep_taylor[.bounded]:** [DeepTaylor](https://www.sciencedirect.com/science/article/pii/S0031320316303582?via%3Dihub) computes for each neuron a root point, that is close to the input, but which's output value is 0, and uses this difference to estimate the attribution of each neuron recursively.
  * **lrp.\*:** [LRP](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) attributes recursively to each neuron's input relevance proportional to its contribution of the neuron output.
  * **integrated_gradients:** [IntegratedGradients](https://arxiv.org/abs/1703.01365) integrates the gradient along a path from the input to a reference.
* *miscellaneous:*
  * **input:** Returns the input.
  * **random:** Returns random Gaussian noise.

**The intention behind iNNvestigate is to make it easy to use analysis methods, but it is not to explain the underlying concepts and assumptions. Please, read the according publication(s) when using a certain method and when publishing please cite the according paper(s) (as well as the [iNNvestigate paper](https://arxiv.org/abs/1808.04260)). Thank you!**

All the available methods have in common that they try to analyze the output of a specific neuron with respect to input to the neural network.
Typically one analyses the neuron with the largest activation in the output layer.
For example, given a Keras model, one can create a 'gradient' analyzer:

```python
import tensorflow as tf
import innvestigate
tf.compat.v1.disable_eager_execution()

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

Let's look at an example ([code](https://github.com/albermax/innvestigate/blob/master/examples/readme_code_snippet.py)) with VGG16 and this image:

![Input image.](https://github.com/albermax/innvestigate/raw/master/examples/images/readme_example_input.png)


```python
import tensorflow as tf
import tensorflow.keras.applications.vgg16 as vgg16
tf.compat.v1.disable_eager_execution()

import innvestigate

# Get model
model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
# Strip softmax layer
model = innvestigate.model_wo_softmax(model)

# Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model)

# Add batch axis and preprocess
x = preprocess(image[None])
# Apply analyzer w.r.t. maximum activated output-neuron
a = analyzer.analyze(x)

# Aggregate along color channels and normalize to [-1, 1]
a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
a /= np.max(np.abs(a))
# Plot
plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
```

![Analysis image.](https://github.com/albermax/innvestigate/raw/master/examples/images/readme_example_analysis.png)

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

**To use the ImageNet examples please download the example images first ([script](https://github.com/albermax/innvestigate/blob/master/examples/images/wget_imagenet_2011_samples.sh)).**

## More documentation

... can be found here:

* Alber, M., Lapuschkin, S., Seegerer, P., Hägele, M., Schütt, K. T., Montavon, G., Samek, W., Müller, K. R., Dähne, S., & Kindermans, P. J. (2019). INNvestigate neural networks! Journal of Machine Learning Research, 20.](https://jmlr.org/papers/v20/18-540.html)
  ```
  @article{JMLR:v20:18-540,
  author  = {Maximilian Alber and Sebastian Lapuschkin and Philipp Seegerer and Miriam H{{\"a}}gele and Kristof T. Sch{{\"u}}tt and Gr{{\'e}}goire Montavon and Wojciech Samek and Klaus-Robert M{{\"u}}ller and Sven D{{\"a}}hne and Pieter-Jan Kindermans},
  title   = {iNNvestigate Neural Networks!},
  journal = {Journal of Machine Learning Research},
  year    = {2019},
  volume  = {20},
  number  = {93},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v20/18-540.html}
  }
  ```
* [https://innvestigate.readthedocs.io/en/latest/](https://innvestigate.readthedocs.io/en/latest/)

## Contributing

If you would like to contribute or add your analysis method 
please open an issue or submit a pull request.

## Releases

[Can be found here.](https://github.com/albermax/innvestigate/blob/master/VERSION.md)


## Acknowledgements

> Adrian Hill acknowledges support by the Federal Ministry of Education and Research (BMBF) for the Berlin Institute for the Foundations of Learning and Data (BIFOLD) (01IS18037A).
