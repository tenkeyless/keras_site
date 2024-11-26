---
title: KerasTuner API
linkTitle: KerasTuner
toc: true
weight: 15
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

The **Hyperparameters** class is used to specify a set of hyperparameters
and their values, to be used in the model building function.

The **Tuner** subclasses corresponding to different tuning algorithms are
called directly by the user to start the search or to get the best models.

The **Oracle** subclasses are the core search algorithms, receiving model evaluation
results from the Tuner and providing new hyperparameter values.

The **HyperModel** subclasses are predefined search spaces for certain model
families like ResNet and XceptionNet.

### [HyperParameters]({{< relref "/docs/api/keras_tuner/hyperparameters/" >}})

- [HyperParameters class]({{< relref "/docs/api/keras_tuner/hyperparameters/#hyperparameters-class" >}})
- [Boolean method]({{< relref "/docs/api/keras_tuner/hyperparameters/#boolean-method" >}})
- [Choice method]({{< relref "/docs/api/keras_tuner/hyperparameters/#choice-method" >}})
- [Fixed method]({{< relref "/docs/api/keras_tuner/hyperparameters/#fixed-method" >}})
- [Float method]({{< relref "/docs/api/keras_tuner/hyperparameters/#float-method" >}})
- [Int method]({{< relref "/docs/api/keras_tuner/hyperparameters/#int-method" >}})
- [conditional\_scope method]({{< relref "/docs/api/keras_tuner/hyperparameters/#conditional_scope-method" >}})
- [get method]({{< relref "/docs/api/keras_tuner/hyperparameters/#get-method" >}})

### [Tuners]({{< relref "/docs/api/keras_tuner/tuners/" >}})

- [The base Tuner class]({{< relref "/docs/api/keras_tuner/tuners/base_tuner" >}})
- [Objective class]({{< relref "/docs/api/keras_tuner/tuners/objective" >}})
- [RandomSearch Tuner]({{< relref "/docs/api/keras_tuner/tuners/random" >}})
- [GridSearch Tuner]({{< relref "/docs/api/keras_tuner/tuners/grid" >}})
- [BayesianOptimization Tuner]({{< relref "/docs/api/keras_tuner/tuners/bayesian" >}})
- [Hyperband Tuner]({{< relref "/docs/api/keras_tuner/tuners/hyperband" >}})
- [Sklearn Tuner]({{< relref "/docs/api/keras_tuner/tuners/sklearn" >}})

### [Oracles]({{< relref "/docs/api/keras_tuner/oracles/" >}})

- [The base Oracle class]({{< relref "/docs/api/keras_tuner/oracles/base_oracle" >}})
- [@synchronized decorator]({{< relref "/docs/api/keras_tuner/oracles/synchronized" >}})
- [RandomSearch Oracle]({{< relref "/docs/api/keras_tuner/oracles/random" >}})
- [GridSearch Oracle]({{< relref "/docs/api/keras_tuner/oracles/grid" >}})
- [BayesianOptimization Oracle]({{< relref "/docs/api/keras_tuner/oracles/bayesian" >}})
- [Hyperband Oracle]({{< relref "/docs/api/keras_tuner/oracles/hyperband" >}})

### [HyperModels]({{< relref "/docs/api/keras_tuner/hypermodels/" >}})

- [The base HyperModel class]({{< relref "/docs/api/keras_tuner/hypermodels/base_hypermodel" >}})
- [HyperEfficientNet]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_efficientnet" >}})
- [HyperImageAugment]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_image_augment" >}})
- [HyperResNet]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_resnet" >}})
- [HyperXception]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_xception" >}})

### [Errors]({{< relref "/docs/api/keras_tuner/errors/" >}})

- [FailedTrialError class]({{< relref "/docs/api/keras_tuner/errors/#failedtrialerror-class" >}})
- [FatalError class]({{< relref "/docs/api/keras_tuner/errors/#fatalerror-class" >}})
- [FatalValueError class]({{< relref "/docs/api/keras_tuner/errors/#fatalvalueerror-class" >}})
- [FatalTypeError class]({{< relref "/docs/api/keras_tuner/errors/#fataltypeerror-class" >}})
- [FatalRuntimeError class]({{< relref "/docs/api/keras_tuner/errors/#fatalruntimeerror-class" >}})
