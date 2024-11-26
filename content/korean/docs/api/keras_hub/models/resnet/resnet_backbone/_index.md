---
title: ResNetBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/resnet/resnet_backbone.py#L10" >}}

### `ResNetBackbone` class

```python
keras_hub.models.ResNetBackbone(
    input_conv_filters,
    input_conv_kernel_sizes,
    stackwise_num_filters,
    stackwise_num_blocks,
    stackwise_num_strides,
    block_type,
    use_pre_activation=False,
    image_shape=(None, None, 3),
    data_format=None,
    dtype=None,
    **kwargs
)
```

ResNet and ResNetV2 core network with hyperparameters.

This class implements a ResNet backbone as described in [Deep Residual
Learning for Image Recognition](https://arxiv.org/abs/1512.03385)(
CVPR 2016), [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)(ECCV 2016), [ResNet strikes back: An
improved training procedure in timm](https://arxiv.org/abs/2110.00476)(
NeurIPS 2021 Workshop) and [Bag of Tricks for Image Classification with
Convolutional Neural Networks](https://arxiv.org/abs/1812.01187).

The difference in ResNet and ResNetV2 rests in the structure of their
individual building blocks. In ResNetV2, the batch normalization and
ReLU activation precede the convolution layers, as opposed to ResNet where
the batch normalization and ReLU activation are applied after the
convolution layers.

ResNetVd introduces two key modifications to the standard ResNet. First,
the initial convolutional layer is replaced by a series of three
successive convolutional layers. Second, shortcut connections use an
additional pooling operation rather than performing downsampling within
the convolutional layers themselves.

**Arguments**

- **input_conv_filters**: list of ints. The number of filters of the initial
  convolution(s).
- **input_conv_kernel_sizes**: list of ints. The kernel sizes of the initial
  convolution(s).
- **stackwise_num_filters**: list of ints. The number of filters for each
  stack.
- **stackwise_num_blocks**: list of ints. The number of blocks for each stack.
- **stackwise_num_strides**: list of ints. The number of strides for each
  stack.
- **block_type**: str. The block type to stack. One of `"basic_block"`,
  `"bottleneck_block"`, `"basic_block_vd"` or
  `"bottleneck_block_vd"`. Use `"basic_block"` for ResNet18 and
  ResNet34. Use `"bottleneck_block"` for ResNet50, ResNet101 and
  ResNet152 and the `"_vd"` prefix for the respective ResNet_vd
  variants.
- **use_pre_activation**: boolean. Whether to use pre-activation or not.
  `True` for ResNetV2, `False` for ResNet.
- **image_shape**: tuple. The input shape without the batch size.
  Defaults to `(None, None, 3)`.
- **data_format**: `None` or str. If specified, either `"channels_last"` or
  `"channels_first"`. The ordering of the dimensions in the
  inputs. `"channels_last"` corresponds to inputs with shape
  `(batch_size, height, width, channels)`
  while `"channels_first"` corresponds to inputs with shape
  `(batch_size, channels, height, width)`. It defaults to the
  `image_data_format` value found in your Keras config file at
  `~/.keras/keras.json`. If you never set it, then it will be
  `"channels_last"`.
- **dtype**: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
  to use for the model's computations and weights.

**Examples**

```python
input_data = np.random.uniform(0, 1, size=(2, 224, 224, 3))
# Pretrained ResNet backbone.
model = keras_hub.models.ResNetBackbone.from_preset("resnet_50_imagenet")
model(input_data)
# Randomly initialized ResNetV2 backbone with a custom config.
model = keras_hub.models.ResNetBackbone(
    input_conv_filters=[64],
    input_conv_kernel_sizes=[7],
    stackwise_num_filters=[64, 64, 64],
    stackwise_num_blocks=[2, 2, 2],
    stackwise_num_strides=[1, 2, 2],
    block_type="basic_block",
    use_pre_activation=True,
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
ResNetBackbone.from_preset(preset, load_weights=True, **kwargs)
```

Instantiate a [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as a
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

This constructor can be called in one of two ways. Either from the base
class like `keras_hub.models.Backbone.from_preset()`, or from
a model class like `keras_hub.models.GemmaBackbone.from_preset()`.
If calling from the base class, the subclass of the returning object
will be inferred from the config in the preset directory.

For any `Backbone` subclass, you can run `cls.presets.keys()` to list
all built-in presets available on the class.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the
  model architecture. If `False`, the weights will be randomly
  initialized.

**Examples**

```python
# Load a Gemma backbone with pre-trained weights.
model = keras_hub.models.Backbone.from_preset(
    "gemma_2b_en",
)
# Load a Bert backbone with a pre-trained config and random weights.
model = keras_hub.models.Backbone.from_preset(
    "bert_base_en",
    load_weights=False,
)
```

| Preset name                       | Parameters | Description                                                                                                                                                                                                        |
| --------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
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
