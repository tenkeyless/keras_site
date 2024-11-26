---
title: KerasTuner Oracles
linkTitle: Oracles
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

The `Oracle` class is the base class for all the search algorithms in KerasTuner.
An `Oracle` object receives evaluation results for a model (from a `Tuner` class)
and generates new hyperparameter values.

The built-in `Oracle` classes are
`RandomSearchOracle`, `BayesianOptimizationOracle`, and `HyperbandOracle`.

You can also write your own tuning algorithm by subclassing the `Oracle` class.

### [The base Oracle class]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/" >}})

- [Oracle class]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#oracle-class" >}})
- [create\_trial function]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#wrapped_func-function" >}})
- [end\_trial function]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#wrapped_func-function" >}})
- [get\_best\_trials method]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#get_best_trials-method" >}})
- [get\_state method]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#get_state-method" >}})
- [set\_state method]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#set_state-method" >}})
- [score\_trial method]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#score_trial-method" >}})
- [populate\_space method]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#populate_space-method" >}})
- [update\_trial function]({{< relref "/docs/api/keras_tuner/oracles/base_oracle/#wrapped_func-function" >}})

### [@synchronized decorator]({{< relref "/docs/api/keras_tuner/oracles/synchronized/" >}})

- [synchronized function]({{< relref "/docs/api/keras_tuner/oracles/synchronized/#synchronized-function" >}})

### [RandomSearch Oracle]({{< relref "/docs/api/keras_tuner/oracles/random/" >}})

- [RandomSearchOracle class]({{< relref "/docs/api/keras_tuner/oracles/random/#randomsearchoracle-class" >}})

### [GridSearch Oracle]({{< relref "/docs/api/keras_tuner/oracles/grid/" >}})

- [GridSearchOracle class]({{< relref "/docs/api/keras_tuner/oracles/grid/#gridsearchoracle-class" >}})

### [BayesianOptimization Oracle]({{< relref "/docs/api/keras_tuner/oracles/bayesian/" >}})

- [BayesianOptimizationOracle class]({{< relref "/docs/api/keras_tuner/oracles/bayesian/#bayesianoptimizationoracle-class" >}})

### [Hyperband Oracle]({{< relref "/docs/api/keras_tuner/oracles/hyperband/" >}})

- [HyperbandOracle class]({{< relref "/docs/api/keras_tuner/oracles/hyperband/#hyperbandoracle-class" >}})
