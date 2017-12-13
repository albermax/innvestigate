# Innvestigate - a toolbox to innvestigate neural network decisions

TODO: UPDATE readme

## Introduction

PatternNet and PatternLRP are methods that help to interpret decision of non-linear neural networks.
They are in a line with the methods DeConvNet, GuidedBackprop and LRP:

![An overview of the different explanation methods.](https://raw.githubusercontent.com/pikinder/nn-patterns/master/images/fig2.png)

and improve on them:

![Different explanation methods on ImageNet.](https://raw.githubusercontent.com/pikinder/nn-patterns/master/images/fig5.png)

For more details we refer to the paper:

```
PatternNet and PatternLRP -- Improving the interpretability of neural networks
Pieter-Jan Kindermans, Kristof T. Schütt, Maximilian Alber, Klaus-Robert Müller, Sven Dähne
https://arxiv.org/abs/1705.05598
```

If you use this code please cite the following paper:
```
TODO: Add link to SW paper.
```


## Installation

To install the code, please clone the repository and run the setup script:

```bash
git clone https://github.com/albermax/innvestigate.git
cd innvestigate
python setup.py install
```

## Usage and Examples

#### Explaining

All the presented methods have in common that they try to explain the output of a specific neuron with respect to input to the neural network.
Typically one explains the neuron with the largest activation in the output layer.
Now given the output layer 'output_layer' of a Lasagne network, one can create an explainer by:

```python
import nn_patterns

output_layer = create_a_lasagne_network()
pattern = load_pattern()

explainer = nn_patterns.create_explainer("patternnet", output_layer, patterns=patterns)
```

and explain the influence of the neural networks input on the output neuron by:

```python
explanation = explainer.explain(input)
```

The following explanation methods are available: 

* *function*:
  * **gradient:** The gradient of the output neuron with respect to the input.
* *signal*:
  * **deconvnet:** [DeConvNet](https://arxiv.org/abs/1311.2901)
  * **guided:** [Guided BackProp](https://arxiv.org/abs/1412.6806)
  * **patternnet:** [PatternNet](https://arxiv.org/abs/1705.05598)
* *interaction*:
  * **patternlrp:** [PatternLRP](https://arxiv.org/abs/1705.05598)
  * **lrp.z:** [LRP](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)

The pattern parameter is only necessary for PatternNet and PatternLRP.

The available options to select the target neuron are:

* "max_output" (default): the neuron with the largest activation.
* integer i: always the neuron at position i.
* None: take the activation of the last layer as they are. This results in a superposition of explanations.


#### Computing patterns

The methods PatternNet and PatternLRP are based on so called patterns that are network and data specific and need to be computed.
Given a training set X and a desired batch_size this can be done in the following way:

```python
import nn_patterns.patterns

computer = nn_patterns.patterns.CombinedPatternComputer(output_layer)
patterns = computer.compute_patterns(X, batch_size=batch_size)
```

#### Examples

In the directory [examples](https://github.com/pikinder/nn-patterns/blob/master/examples/) one can find different examples as Python scripts and as Jupyter notebooks:

* **[step_by_step_cifar10]():** explains how to compute patterns for a given neural networks and how to use them with PatternNet and PatternLRP.
* **[step_by_step_imagenet](https://github.com/pikinder/nn-patterns/blob/master/examples/step_by_step_imagenet.ipynb):** explains how to apply pre-computed patterns for the VGG16 network on ImageNet.
* **[all_methods](https://github.com/pikinder/nn-patterns/blob/master/examples/all_methods.ipynb):** shows how to use the different methods with VGG16 on ImageNet, i.e. the reproduce the explanation grid above.
