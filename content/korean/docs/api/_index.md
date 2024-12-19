---
title: Keras 3 API 문서
linkTitle: Keras 3 API 문서
toc: true
weight: 5
---

{{< keras/original checkedAt="2024-11-26" >}}

### [Models API]({{< relref "/docs/api/models/" >}})

- [The Model class]({{< relref "/docs/api/models/model" >}})
- [The Sequential class]({{< relref "/docs/api/models/sequential" >}})
- [Model training APIs]({{< relref "/docs/api/models/model_training_apis" >}})
- [Saving & serialization]({{< relref "/docs/api/models/model_saving_apis/" >}})

### [Layers API]({{< relref "/docs/api/layers/" >}})

- [The base Layer class]({{< relref "/docs/api/layers/base_layer" >}})
- [Layer activations]({{< relref "/docs/api/layers/activations" >}})
- [Layer weight initializers]({{< relref "/docs/api/layers/initializers" >}})
- [Layer weight regularizers]({{< relref "/docs/api/layers/regularizers" >}})
- [Layer weight constraints]({{< relref "/docs/api/layers/constraints" >}})
- [Core layers]({{< relref "/docs/api/layers/core_layers/" >}})
- [Convolution layers]({{< relref "/docs/api/layers/convolution_layers/" >}})
- [Pooling layers]({{< relref "/docs/api/layers/pooling_layers/" >}})
- [Recurrent layers]({{< relref "/docs/api/layers/recurrent_layers/" >}})
- [Preprocessing layers]({{< relref "/docs/api/layers/preprocessing_layers/" >}})
- [Normalization layers]({{< relref "/docs/api/layers/normalization_layers/" >}})
- [Regularization layers]({{< relref "/docs/api/layers/regularization_layers/" >}})
- [Attention layers]({{< relref "/docs/api/layers/attention_layers/" >}})
- [Reshaping layers]({{< relref "/docs/api/layers/reshaping_layers/" >}})
- [Merging layers]({{< relref "/docs/api/layers/merging_layers/" >}})
- [Activation layers]({{< relref "/docs/api/layers/activation_layers/" >}})
- [Backend-specific layers]({{< relref "/docs/api/layers/backend_specific_layers/" >}})

### [Callbacks API]({{< relref "/docs/api/callbacks/" >}})

- [Base Callback class]({{< relref "/docs/api/callbacks/base_callback" >}})
- [ModelCheckpoint]({{< relref "/docs/api/callbacks/model_checkpoint" >}})
- [BackupAndRestore]({{< relref "/docs/api/callbacks/backup_and_restore" >}})
- [TensorBoard]({{< relref "/docs/api/callbacks/tensorboard" >}})
- [EarlyStopping]({{< relref "/docs/api/callbacks/early_stopping" >}})
- [LearningRateScheduler]({{< relref "/docs/api/callbacks/learning_rate_scheduler" >}})
- [ReduceLROnPlateau]({{< relref "/docs/api/callbacks/reduce_lr_on_plateau" >}})
- [RemoteMonitor]({{< relref "/docs/api/callbacks/remote_monitor" >}})
- [LambdaCallback]({{< relref "/docs/api/callbacks/lambda_callback" >}})
- [TerminateOnNaN]({{< relref "/docs/api/callbacks/terminate_on_nan" >}})
- [CSVLogger]({{< relref "/docs/api/callbacks/csv_logger" >}})
- [ProgbarLogger]({{< relref "/docs/api/callbacks/progbar_logger" >}})
- [SwapEMAWeights]({{< relref "/docs/api/callbacks/swap_ema_weights" >}})

### [Ops API]({{< relref "/docs/api/ops/" >}})

- [NumPy ops]({{< relref "/docs/api/ops/numpy/" >}})
- [NN ops]({{< relref "/docs/api/ops/nn/" >}})
- [Linear algebra ops]({{< relref "/docs/api/ops/linalg/" >}})
- [Core ops]({{< relref "/docs/api/ops/core/" >}})
- [Image ops]({{< relref "/docs/api/ops/image/" >}})
- [FFT ops]({{< relref "/docs/api/ops/fft/" >}})

### [Optimizers]({{< relref "/docs/api/optimizers/" >}})

- [SGD]({{< relref "/docs/api/optimizers/sgd" >}})
- [RMSprop]({{< relref "/docs/api/optimizers/rmsprop" >}})
- [Adam]({{< relref "/docs/api/optimizers/adam" >}})
- [AdamW]({{< relref "/docs/api/optimizers/adamw" >}})
- [Adadelta]({{< relref "/docs/api/optimizers/adadelta" >}})
- [Adagrad]({{< relref "/docs/api/optimizers/adagrad" >}})
- [Adamax]({{< relref "/docs/api/optimizers/adamax" >}})
- [Adafactor]({{< relref "/docs/api/optimizers/adafactor" >}})
- [Nadam]({{< relref "/docs/api/optimizers/Nadam" >}})
- [Ftrl]({{< relref "/docs/api/optimizers/ftrl" >}})
- [Lion]({{< relref "/docs/api/optimizers/lion" >}})
- [Lamb]({{< relref "/docs/api/optimizers/lamb" >}})
- [Loss Scale Optimizer]({{< relref "/docs/api/optimizers/loss_scale_optimizer" >}})

### [Metrics]({{< relref "/docs/api/metrics/" >}})

- [Base Metric class]({{< relref "/docs/api/metrics/base_metric" >}})
- [Accuracy metrics]({{< relref "/docs/api/metrics/accuracy_metrics" >}})
- [Probabilistic metrics]({{< relref "/docs/api/metrics/probabilistic_metrics" >}})
- [Regression metrics]({{< relref "/docs/api/metrics/regression_metrics" >}})
- [Classification metrics based on True/False positives & negatives]({{< relref "/docs/api/metrics/classification_metrics" >}})
- [Image segmentation metrics]({{< relref "/docs/api/metrics/segmentation_metrics" >}})
- [Hinge metrics for "maximum-margin" classification]({{< relref "/docs/api/metrics/hinge_metrics" >}})
- [Metric wrappers and reduction metrics]({{< relref "/docs/api/metrics/metrics_wrappers" >}})

### [Losses]({{< relref "/docs/api/losses/" >}})

- [Probabilistic losses]({{< relref "/docs/api/losses/probabilistic_losses" >}})
- [Regression losses]({{< relref "/docs/api/losses/regression_losses" >}})
- [Hinge losses for "maximum-margin" classification]({{< relref "/docs/api/losses/hinge_losses" >}})

### [Data loading]({{< relref "/docs/api/data_loading/" >}})

- [Image data loading]({{< relref "/docs/api/data_loading/image" >}})
- [Timeseries data loading]({{< relref "/docs/api/data_loading/timeseries" >}})
- [Text data loading]({{< relref "/docs/api/data_loading/text" >}})
- [Audio data loading]({{< relref "/docs/api/data_loading/audio" >}})

### [Built-in small datasets]({{< relref "/docs/api/datasets/" >}})

- [MNIST digits classification dataset]({{< relref "/docs/api/datasets/mnist" >}})
- [CIFAR10 small images classification dataset]({{< relref "/docs/api/datasets/cifar10" >}})
- [CIFAR100 small images classification dataset]({{< relref "/docs/api/datasets/cifar100" >}})
- [IMDB movie review sentiment classification dataset]({{< relref "/docs/api/datasets/imdb" >}})
- [Reuters newswire classification dataset]({{< relref "/docs/api/datasets/reuters" >}})
- [Fashion MNIST dataset, an alternative to MNIST]({{< relref "/docs/api/datasets/fashion_mnist" >}})
- [California Housing price regression dataset]({{< relref "/docs/api/datasets/california_housing" >}})

### [Keras Applications]({{< relref "/docs/api/applications/" >}})

- [Xception]({{< relref "/docs/api/applications/xception" >}})
- [EfficientNet B0 to B7]({{< relref "/docs/api/applications/efficientnet" >}})
- [EfficientNetV2 B0 to B3 and S, M, L]({{< relref "/docs/api/applications/efficientnet_v2" >}})
- [ConvNeXt Tiny, Small, Base, Large, XLarge]({{< relref "/docs/api/applications/convnext" >}})
- [VGG16 and VGG19]({{< relref "/docs/api/applications/vgg" >}})
- [ResNet and ResNetV2]({{< relref "/docs/api/applications/resnet" >}})
- [MobileNet, MobileNetV2, and MobileNetV3]({{< relref "/docs/api/applications/mobilenet" >}})
- [DenseNet]({{< relref "/docs/api/applications/densenet" >}})
- [NasNetLarge and NasNetMobile]({{< relref "/docs/api/applications/nasnet" >}})
- [InceptionV3]({{< relref "/docs/api/applications/inceptionv3" >}})
- [InceptionResNetV2]({{< relref "/docs/api/applications/inceptionresnetv2" >}})

### [Mixed precision]({{< relref "/docs/api/mixed_precision/" >}})

- [Mixed precision policy API]({{< relref "/docs/api/mixed_precision/policy" >}})

### [Multi-device distribution]({{< relref "/docs/api/distribution/" >}})

- [LayoutMap API]({{< relref "/docs/api/distribution/layout_map" >}})
- [DataParallel API]({{< relref "/docs/api/distribution/data_parallel" >}})
- [ModelParallel API]({{< relref "/docs/api/distribution/model_parallel" >}})
- [ModelParallel API]({{< relref "/docs/api/distribution/model_parallel" >}})
- [Distribution utilities]({{< relref "/docs/api/distribution/distribution_utils" >}})

### [RNG API]({{< relref "/docs/api/random/" >}})

- [SeedGenerator class]({{< relref "/docs/api/random/seed_generator" >}})
- [Random operations]({{< relref "/docs/api/random/random_ops" >}})

### [Utilities]({{< relref "/docs/api/utils/" >}})

- [Experiment management utilities]({{< relref "/docs/api/utils/experiment_management_utils" >}})
- [Model plotting utilities]({{< relref "/docs/api/utils/model_plotting_utils" >}})
- [Structured data preprocessing utilities]({{< relref "/docs/api/utils/feature_space" >}})
- [Tensor utilities]({{< relref "/docs/api/utils/tensor_utils" >}})
- [Python & NumPy utilities]({{< relref "/docs/api/utils/python_utils" >}})
- [Keras configuration utilities]({{< relref "/docs/api/utils/config_utils" >}})

### [KerasTuner]({{< relref "/docs/api/keras_tuner/" >}})

- [HyperParameters]({{< relref "/docs/api/keras_tuner/hyperparameters" >}})
- [Tuners]({{< relref "/docs/api/keras_tuner/tuners/" >}})
- [Oracles]({{< relref "/docs/api/keras_tuner/oracles/" >}})
- [HyperModels]({{< relref "/docs/api/keras_tuner/hypermodels/" >}})
- [Errors]({{< relref "/docs/api/keras_tuner/errors" >}})

### [KerasCV]({{< relref "/docs/api/keras_cv/" >}})

- [Layers]({{< relref "/docs/api/keras_cv/layers/" >}})
- [Models]({{< relref "/docs/api/keras_cv/models/" >}})
- [Bounding box formats and utilities]({{< relref "/docs/api/keras_cv/bounding_box/" >}})
- [Losses]({{< relref "/docs/api/keras_cv/losses/" >}})

### [KerasNLP]({{< relref "/docs/api/keras_nlp/" >}})

- [Pretrained Models]({{< relref "/docs/api/keras_nlp/models/" >}})
- [Models API]({{< relref "/docs/api/keras_nlp/base_classes/" >}})
- [Tokenizers]({{< relref "/docs/api/keras_nlp/tokenizers/" >}})
- [Preprocessing Layers]({{< relref "/docs/api/keras_nlp/preprocessing_layers/" >}})
- [Modeling Layers]({{< relref "/docs/api/keras_nlp/modeling_layers/" >}})
- [Samplers]({{< relref "/docs/api/keras_nlp/samplers/" >}})
- [Metrics]({{< relref "/docs/api/keras_nlp/metrics/" >}})

### [KerasHub]({{< relref "/docs/api/keras_hub/" >}})

- [Pretrained Models]({{< relref "/docs/api/keras_hub/models/" >}})
- [Models API]({{< relref "/docs/api/keras_hub/base_classes/" >}})
- [Tokenizers]({{< relref "/docs/api/keras_hub/tokenizers/" >}})
- [Preprocessing Layers]({{< relref "/docs/api/keras_hub/preprocessing_layers/" >}})
- [Modeling Layers]({{< relref "/docs/api/keras_hub/modeling_layers/" >}})
- [Samplers]({{< relref "/docs/api/keras_hub/samplers/" >}})
- [Metrics]({{< relref "/docs/api/keras_hub/metrics/" >}})
