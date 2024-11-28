---
title: 개발자 가이드
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

Our developer guides are deep-dives into specific topics such as layer subclassing, fine-tuning, or model saving. They're one of the best ways to become a Keras expert.

Most of our guides are written as Jupyter notebooks and can be run in one click in [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb), a hosted notebook environment that requires no setup and runs in the cloud. Google Colab includes GPU and TPU runtimes.

## Available guides

- [The Functional API]({{< relref "/docs/guides/functional_api" >}})
- [The Sequential model]({{< relref "/docs/guides/sequential_model" >}})
- [Making new layers & models via subclassing]({{< relref "/docs/guides/making_new_layers_and_models_via_subclassing" >}})
- [Training & evaluation with the built-in methods]({{< relref "/docs/guides/training_with_built_in_methods" >}})
- [Customizing `fit()` with JAX]({{< relref "/docs/guides/custom_train_step_in_jax" >}})
- [Customizing `fit()` with TensorFlow]({{< relref "/docs/guides/custom_train_step_in_tensorflow" >}})
- [Customizing `fit()` with PyTorch]({{< relref "/docs/guides/custom_train_step_in_torch" >}})
- [Writing a custom training loop in JAX]({{< relref "/docs/guides/writing_a_custom_training_loop_in_jax" >}})
- [Writing a custom training loop in TensorFlow]({{< relref "/docs/guides/writing_a_custom_training_loop_in_tensorflow" >}})
- [Writing a custom training loop in PyTorch]({{< relref "/docs/guides/writing_a_custom_training_loop_in_torch" >}})
- [Serialization & saving]({{< relref "/docs/guides/serialization_and_saving" >}})
- [Customizing saving & serialization]({{< relref "/docs/guides/customizing_saving_and_serialization" >}})
- [Writing your own callbacks]({{< relref "/docs/guides/writing_your_own_callbacks" >}})
- [Transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning" >}})
- [Distributed training with JAX]({{< relref "/docs/guides/distributed_training_with_jax" >}})
- [Distributed training with TensorFlow]({{< relref "/docs/guides/distributed_training_with_tensorflow" >}})
- [Distributed training with PyTorch]({{< relref "/docs/guides/distributed_training_with_torch" >}})
- [Distributed training with Keras 3]({{< relref "/docs/guides/distribution" >}})
- [Migrating Keras 2 code to Keras 3]({{< relref "/docs/guides/migrating_to_keras_3" >}})

### [Hyperparameter Tuning]({{< relref "/docs/guides/keras_tuner" >}})

- [Getting started with KerasTuner]({{< relref "/docs/guides/keras_tuner/getting_started" >}})
- [Distributed hyperparameter tuning with KerasTuner]({{< relref "/docs/guides/keras_tuner/distributed_tuning" >}})
- [Tune hyperparameters in your custom training loop]({{< relref "/docs/guides/keras_tuner/custom_tuner" >}})
- [Visualize the hyperparameter tuning process]({{< relref "/docs/guides/keras_tuner/visualize_tuning" >}})
- [Handling failed trials in KerasTuner]({{< relref "/docs/guides/keras_tuner/failed_trials" >}})
- [Tailor the search space]({{< relref "/docs/guides/keras_tuner/tailor_the_search_space" >}})

### [KerasCV]({{< relref "/docs/guides/keras_cv" >}})

- [Use KerasCV to assemble object detection pipelines]({{< relref "/docs/guides/keras_cv/object_detection_keras_cv" >}})
- [Use KerasCV to train powerful image classifiers.]({{< relref "/docs/guides/keras_cv/classification_with_keras_cv" >}})
- [CutMix, MixUp, and RandAugment image augmentation with KerasCV]({{< relref "/docs/guides/keras_cv/cut_mix_mix_up_and_rand_augment" >}})
- [High-performance image generation using Stable Diffusion in KerasCV]({{< relref "/docs/guides/keras_cv/generate_images_with_stable_diffusion" >}})
- [Custom Image Augmentations with BaseImageAugmentationLayer]({{< relref "/docs/guides/keras_cv/custom_image_augmentations" >}})
- [Semantic Segmentation with KerasCV]({{< relref "/docs/guides/keras_cv/semantic_segmentation_deeplab_v3_plus" >}})
- [Segment Anything in KerasCV]({{< relref "/docs/guides/keras_cv/segment_anything_in_keras_cv" >}})

### [KerasNLP]({{< relref "/docs/guides/keras_nlp" >}})

- [Getting Started with KerasNLP]({{< relref "/docs/guides/keras_nlp/getting_started" >}})
- [Pretraining a Transformer from scratch with KerasNLP]({{< relref "/docs/guides/keras_nlp/transformer_pretraining" >}})
- [Uploading Models with KerasNLP]({{< relref "/docs/guides/keras_nlp/upload" >}})

### [KerasHub]({{< relref "/docs/guides/keras_hub" >}})

- [Getting Started with KerasHub]({{< relref "/docs/guides/keras_hub/getting_started" >}})
- [Semantic Segmentation with KerasHub]({{< relref "/docs/guides/keras_hub/semantic_segmentation_deeplab_v3" >}})
- [Pretraining a Transformer from scratch with KerasHub]({{< relref "/docs/guides/keras_hub/transformer_pretraining" >}})
- [Uploading Models with KerasHub]({{< relref "/docs/guides/keras_hub/upload" >}})
- [Classification with KerasHub]({{< relref "/docs/guides/keras_hub/classification_with_keras_hub" >}})
- [Segment Anything in KerasHub]({{< relref "/docs/guides/keras_hub/segment_anything_in_keras_hub" >}})
- [Stable Diffusion 3 in KerasHub]({{< relref "/docs/guides/keras_hub/stable_diffusion_3_in_keras_hub" >}})
