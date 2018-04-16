# iNNvestigate neural networks!

## Note: The library is in under alpha testing! Please let us know if you find any bugs.

## Introduction

In the recent years neural networks furthered the state of the art in many domains like, e.g., object detection and speech recognition.
Despite the success neural networks are typically still treated as black boxes. Their internal workings are not fully understood and the basis for their predictions is unclear.
In the attempt to understand neural networks better several methods were propose, a.i, Saliency, Deconvnet, GuidedBackprop, SmoothGrad, IntergratedGradients, LRP, PatternNet-\&Attribution.
None of the methods is fully solves the stated problems and due to the lack of a reference implementations their comparison is linked to a major effort.
This library addresses this by providing a common interface and out-of-the-box implementation for many analysis methods.
Our goal is to make analyzing neural network's predictions easy!

# TODO Add picture from notebook here..

![Different explanation methods on ImageNet.](https://raw.githubusercontent.com/pikinder/nn-patterns/master/images/fig5.png)


**If you use this code please star the repository and cite the following paper:**
```
TODO: Add link to SW paper.
```

## Installation

iNNvestigate can be installed with the following command:

```bash
pip install git+https://github.com/albermax/innvestigate
```

To use the example scripts and notebooks one additionally needs to install the package matplotlib:

```bash
pip install matplotlib
```

The tests of the library can be executed via:
```bash
git clone https://github.com/albermax/innvestigate.git
cd innvestigate
python setup.py test
```

The library was developed and tested on a Linux platform with Python 3.5 and Cuda 8.x. Currently only the Keras Tensorflow backend is supported.

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
  * **lrp.\*:** *coming soon* [LRP](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
* *miscellaneous:*
  * **input:** Returns the input.
  * **random:** Returns random Gaussian noise.

All the available methods have in common that they try to analyze the output of a specific neuron with respect to input to the neural network.
Typically one analysis the neuron with the largest activation in the output layer.
Now for example, given a Keras model, one can create a 'gradient' analyzer:

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

analyzer = innvestigate.create_analyzer("pattern.net")
analyzer.fit(X_train)
analysis = analyzer.analyze(X_test)
```

#### Examples

In the directory [examples](https://github.com/albermax/innvestigate/blob/master/examples/) one can find different examples as Python scripts and as Jupyter notebooks:

# TODO: update this list.
* **[step_by_step_cifar10]():** explains how to compute patterns for a given neural networks and how to use them with PatternNet and PatternLRP.
* **[step_by_step_imagenet](https://github.com/pikinder/nn-patterns/blob/master/examples/step_by_step_imagenet.ipynb):** explains how to apply pre-computed patterns for the VGG16 network on ImageNet.
* **[all_methods](https://github.com/pikinder/nn-patterns/blob/master/examples/all_methods.ipynb):** shows how to use the different methods with VGG16 on ImageNet, i.e. the reproduce the explanation grid above.


## Contribution

If you would like to add your analysis method please get in touch with us!

## Version history

[Can be found here.](https://github.com/albermax/innvestigate/blob/master/VERSION.md)