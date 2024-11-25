---
title: The Tuner classes in KerasTuner
linkTitle: Tuners
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

The base `Tuner` class is the class that manages the hyperparameter search process,
including model creation, training, and evaluation. For each trial, a `Tuner` receives new
hyperparameter values from an `Oracle` instance. After calling `model.fit(...)`, it
sends the evaluation results back to the `Oracle` instance and it retrieves the next set
of hyperparameters to try.

There are a few built-in `Tuner` subclasses available for widely-used tuning
algorithms: `RandomSearch`, `BayesianOptimization` and `Hyperband`.

You can also subclass the `Tuner` class to customize your tuning process.
In particular, you can [override the `run_trial` function]({{< relref "/docs/guides/keras_tuner/custom_tuner/#overriding-runtrial" >}})
to customize model building and training.

### [The base Tuner class]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/" >}})

- [Tuner class]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#tuner-class" >}})
- [get\_best\_hyperparameters method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#get_best_hyperparameters-method" >}})
- [get\_best\_models method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#get_best_models-method" >}})
- [get\_state method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#get_state-method" >}})
- [load\_model method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#load_model-method" >}})
- [on\_epoch\_begin method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#on_epoch_begin-method" >}})
- [on\_batch\_begin method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#on_batch_begin-method" >}})
- [on\_batch\_end method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#on_batch_end-method" >}})
- [on\_epoch\_end method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#on_epoch_end-method" >}})
- [run\_trial method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#run_trial-method" >}})
- [results\_summary method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#results_summary-method" >}})
- [save\_model method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#save_model-method" >}})
- [search method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#search-method" >}})
- [search\_space\_summary method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#search_space_summary-method" >}})
- [set\_state method]({{< relref "/docs/api/keras_tuner/tuners/base_tuner/#set_state-method" >}})

### [Objective class]({{< relref "/docs/api/keras_tuner/tuners/objective/" >}})

- [Objective class]({{< relref "/docs/api/keras_tuner/tuners/objective/#objective-class" >}})

### [RandomSearch Tuner]({{< relref "/docs/api/keras_tuner/tuners/random/" >}})

- [RandomSearch class]({{< relref "/docs/api/keras_tuner/tuners/random/#randomsearch-class" >}})

### [GridSearch Tuner]({{< relref "/docs/api/keras_tuner/tuners/grid/" >}})

- [GridSearch class]({{< relref "/docs/api/keras_tuner/tuners/grid/#gridsearch-class" >}})

### [BayesianOptimization Tuner]({{< relref "/docs/api/keras_tuner/tuners/bayesian/" >}})

- [BayesianOptimization class]({{< relref "/docs/api/keras_tuner/tuners/bayesian/#bayesianoptimization-class" >}})

### [Hyperband Tuner]({{< relref "/docs/api/keras_tuner/tuners/hyperband/" >}})

- [Hyperband class]({{< relref "/docs/api/keras_tuner/tuners/hyperband/#hyperband-class" >}})

### [Sklearn Tuner]({{< relref "/docs/api/keras_tuner/tuners/sklearn/" >}})

- [SklearnTuner class]({{< relref "/docs/api/keras_tuner/tuners/sklearn/#sklearntuner-class" >}})
