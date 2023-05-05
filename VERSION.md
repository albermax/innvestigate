<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
![Documentation][badge-docs]
-->
# iNNvestigate Changelog
## Version 2.1.0
- ![Feature][badge-feature] Add way to strip arbitrary output activations ([#310](https://github.com/albermax/innvestigate/pull/310))
- ![Bugfix][badge-bugfix] Fix `is_layer_at_idx` for LRP ([#308](https://github.com/albermax/innvestigate/pull/308))
- ![Maintenance][badge-maintenance] Support Python 3.11 and TensorFlow 2.12 ([#313](https://github.com/albermax/innvestigate/pull/313))
- ![Maintenance][badge-maintenance] Update CI workflows and dev-dependencies ([#313](https://github.com/albermax/innvestigate/pull/313))

## Version 2.0.2
- ![Bugfix][badge-bugfix] Fix `Perturbate` on RGB images ([#306](https://github.com/albermax/innvestigate/pull/306))
- ![Bugfix][badge-bugfix] Fix documentation
- ![Maintenance][badge-maintenance] Support TensorFlow 2.11 ([#300](https://github.com/albermax/innvestigate/pull/300))
- ![Maintenance][badge-maintenance] Update CI workflows and dev-dependencies ([#301](https://github.com/albermax/innvestigate/pull/301))
## Version 2.0.1
- ![Maintenance][badge-maintenance] Remove dead `analyzer.fit` code left from PatternNet and PatternAttribution ([#289](https://github.com/albermax/innvestigate/pull/289))
- ![Documentation][badge-docs] Fixes to README and documentation

## Version 2.0.0
iNNvestigate for TensorFlow 2. This is a [major version release](https://semver.org) and therefore breaking backward compatibility.

Breaking changes:
- update lower dependency bounds to Python 3.8 and TensorFlow 2.6
- use TensorFlow's Keras instead of deprecated stand-alone Keras
- manual disabling of eager execution is required via `tf.compat.v1.disable_eager_execution()` ([#277](https://github.com/albermax/innvestigate/pull/277))
- temporarily remove `PatternNet`, `PatternAttribution`, `LRPZIgnoreBias` and `LRPEpsilonIgnoreBias` ([#277](https://github.com/albermax/innvestigate/pull/277))
- remove DeepLIFT ([#257](https://github.com/albermax/innvestigate/pull/257))

Changes for developers:
- switch setup to Poetry ([#257](https://github.com/albermax/innvestigate/pull/257))
- adopt `src` and `tests` layout ([#257](https://github.com/albermax/innvestigate/pull/257))
- adopt Black code style ([#247](https://github.com/albermax/innvestigate/pull/247))
- add linters to dev dependencies ([#257](https://github.com/albermax/innvestigate/pull/257))
- added type annotations ([#263](https://github.com/albermax/innvestigate/pull/263), [#266](https://github.com/albermax/innvestigate/pull/266), [#277](https://github.com/albermax/innvestigate/pull/277))
- added reference tests & CI to guarantee identical attributions compared to `v1.0.9`  ([#258](https://github.com/albermax/innvestigate/pull/258), [#277](https://github.com/albermax/innvestigate/pull/277))
- refactor backend ([#263](https://github.com/albermax/innvestigate/pull/263), [#277](https://github.com/albermax/innvestigate/pull/277))
- refactor analyzers: explicit class attributes, fixes for serialization ([#266](https://github.com/albermax/innvestigate/pull/266), [#277](https://github.com/albermax/innvestigate/pull/277))
- bug fixes ([#263](https://github.com/albermax/innvestigate/pull/263))

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

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-security]: https://img.shields.io/badge/security-black.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg