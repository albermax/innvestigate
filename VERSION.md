## Version 1.10.0
Spring cleaning release in preparation of iNNvestigate 2.0:
- switch setup to Poetry
- adopt `src` layout
- move tests to separate `tests` folder
- format code with Black
- update dependencies to Python 3.7, TF 1.15 and Keras 2.3
- setup tox

Backwards compatibility breaking changes:
- remove DeepLIFT

## Version 1.0.9

- BatchNormalization Layer compatible with LRP
- EmbeddingLayer support
- new Alpha-Beta-LRP-rules

Additionally various PR were merged and bugs fixed,
for details see [PR #222](https://github.com/albermax/innvestigate/pull/222)

## Version 1.0.8

Bugfixes, increased code coverage, CI.

## Version 1.0.7

Add Python 2 compatibility again.

Bugfixes.

## Version 1.0.6

* Add beta version of DeepLIFT (as in Ancona et.al.) and wrapper for DeepLIFT package.
* Updating readme and bugfixes.

## Version 1.0.5

Treat IntegratedGradients as attribution method and bugfixes.

## Version 1.0.1-1.0.4

Added the following functionality:

* Additional notebooks.
* Analyzers: Input\*Gradient
* Added parameter to choose between plain, abs, square gradient in Gradient analyzer.
* New interface via register-methods in analyzer base code.
* Many fixes.
* Support for read-the-docs documentation.

## Version 1.0.0

Includes the following functionality:

* Analyzers: Gradient, SmoothGrad, IntegratedGradients, PatternNet, PatternAttribution, LRP, DeepTaylor, Input, Random.
* Pattern computer.
