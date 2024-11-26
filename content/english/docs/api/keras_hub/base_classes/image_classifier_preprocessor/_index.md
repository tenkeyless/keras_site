---
title: ImageClassifierPreprocessor
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/image_classifier_preprocessor.py#L8" >}}

### `ImageClassifierPreprocessor` class

```python
keras_hub.models.ImageClassifierPreprocessor(image_converter=None, **kwargs)
```

Base class for image classification preprocessing layers.

`ImageClassifierPreprocessor` tasks wraps a
[`keras_hub.layers.ImageConverter`]({{< relref "/docs/api/keras_hub/preprocessing_layers/image_converter#imageconverter-class" >}}) to create a preprocessing layer for
image classification tasks. It is intended to be paired with a
[`keras_hub.models.ImageClassifier`]({{< relref "/docs/api/keras_hub/base_classes/image_classifier#imageclassifier-class" >}}) task.

All `ImageClassifierPreprocessor` take inputs three inputs, `x`, `y`, and
`sample_weight`. `x`, the first input, should always be included. It can
be a image or batch of images. See examples below. `y` and `sample_weight`
are optional inputs that will be passed through unaltered. Usually, `y` will
be the classification label, and `sample_weight` will not be provided.

The layer will output either `x`, an `(x, y)` tuple if labels were provided,
or an `(x, y, sample_weight)` tuple if labels and sample weight were
provided. `x` will be the input images after all model preprocessing has
been applied.

All `ImageClassifierPreprocessor` tasks include a `from_preset()`
constructor which can be used to load a pre-trained config and vocabularies.
You can call the `from_preset()` constructor directly on this base class, in
which case the correct class for your model will be automatically
instantiated.

Examples.

```python
preprocessor = keras_hub.models.ImageClassifierPreprocessor.from_preset(
    "resnet_50",
)
# Resize a single image for resnet 50.
x = np.random.randint(0, 256, (512, 512, 3))
x = preprocessor(x)
# Resize a labeled image.
x, y = np.random.randint(0, 256, (512, 512, 3)), 1
x, y = preprocessor(x, y)
# Resize a batch of labeled images.
x, y = [np.random.randint(0, 256, (512, 512, 3)), np.zeros((512, 512, 3))], [1, 0]
x, y = preprocessor(x, y)
# Use a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
ImageClassifierPreprocessor.from_preset(
    preset, config_file="preprocessor.json", **kwargs
)
```

Instantiate a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Preprocessor` subclass, you can run `cls.presets.keys()` to
list all built-in presets available on the class.

As there are usually multiple preprocessing classes for a given model,
this method should be called on a specific subclass like
`keras_hub.models.BertTextClassifierPreprocessor.from_preset()`.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.

**Examples**

```python
# Load a preprocessor for Gemma generation.
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en",
)
# Load a preprocessor for Bert classification.
preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en",
)
```

| Preset name                       | Parameters | Description                                                                                                                                                                                                        |
| --------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| mit_b0_ade20k_512                 | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks.                                                                                                                                                              |
| mit_b1_ade20k_512                 | 13.16M     | MiT (MixTransformer) model with 8 transformer blocks.                                                                                                                                                              |
| mit_b2_ade20k_512                 | 24.20M     | MiT (MixTransformer) model with 16 transformer blocks.                                                                                                                                                             |
| mit_b3_ade20k_512                 | 44.08M     | MiT (MixTransformer) model with 28 transformer blocks.                                                                                                                                                             |
| mit_b4_ade20k_512                 | 60.85M     | MiT (MixTransformer) model with 41 transformer blocks.                                                                                                                                                             |
| mit_b5_ade20k_640                 | 81.45M     | MiT (MixTransformer) model with 52 transformer blocks.                                                                                                                                                             |
| mit_b0_cityscapes_1024            | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks.                                                                                                                                                              |
| mit_b1_cityscapes_1024            | 13.16M     | MiT (MixTransformer) model with 8 transformer blocks.                                                                                                                                                              |
| mit_b2_cityscapes_1024            | 24.20M     | MiT (MixTransformer) model with 16 transformer blocks.                                                                                                                                                             |
| mit_b3_cityscapes_1024            | 44.08M     | MiT (MixTransformer) model with 28 transformer blocks.                                                                                                                                                             |
| mit_b4_cityscapes_1024            | 60.85M     | MiT (MixTransformer) model with 41 transformer blocks.                                                                                                                                                             |
| mit_b5_cityscapes_1024            | 81.45M     | MiT (MixTransformer) model with 52 transformer blocks.                                                                                                                                                             |
| resnet_18_imagenet                | 11.19M     | 18-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                              |
| resnet_50_imagenet                | 23.56M     | 50-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                              |
| resnet_101_imagenet               | 42.61M     | 101-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                             |
| resnet_152_imagenet               | 58.30M     | 152-layer ResNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                             |
| resnet_v2_50_imagenet             | 23.56M     | 50-layer ResNetV2 model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                            |
| resnet_v2_101_imagenet            | 42.61M     | 101-layer ResNetV2 model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                           |
| resnet_vd_18_imagenet             | 11.72M     | 18-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                |
| resnet_vd_34_imagenet             | 21.84M     | 34-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                |
| resnet_vd_50_imagenet             | 25.63M     | 50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                |
| resnet_vd_50_ssld_imagenet        | 25.63M     | 50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation.                                                                    |
| resnet_vd_50_ssld_v2_imagenet     | 25.63M     | 50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation and AutoAugment.                                                    |
| resnet_vd_50_ssld_v2_fix_imagenet | 25.63M     | 50-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation, AutoAugment and additional fine-tuning of the classification head. |
| resnet_vd_101_imagenet            | 44.67M     | 101-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                               |
| resnet_vd_101_ssld_imagenet       | 44.67M     | 101-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution with knowledge distillation.                                                                   |
| resnet_vd_152_imagenet            | 60.36M     | 152-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                               |
| resnet_vd_200_imagenet            | 74.93M     | 200-layer ResNetVD (ResNet with bag of tricks) model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                               |
| densenet_121_imagenet             | 7.04M      | 121-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                           |
| densenet_169_imagenet             | 12.64M     | 169-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                           |
| densenet_201_imagenet             | 18.32M     | 201-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                           |
| vgg_11_imagenet                   | 9.22M      | 11-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                                 |
| vgg_13_imagenet                   | 9.40M      | 13-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                                 |
| vgg_16_imagenet                   | 14.71M     | 16-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                                 |
| vgg_19_imagenet                   | 20.02M     | 19-layer vgg model pre-trained on the ImageNet 1k dataset at a 224x224 resolution.                                                                                                                                 |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L238" >}}

### `save_to_preset` method

```python
ImageClassifier.save_to_preset(preset_dir)
```

Save task to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.
