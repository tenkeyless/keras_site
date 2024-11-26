---
title: MobileNetV3 backbones
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mobilenet_v3/mobilenet_v3_backbone.py#L43" >}}

### `MobileNetV3Backbone` class

```python
keras_cv.models.MobileNetV3Backbone(
    stackwise_expansion,
    stackwise_filters,
    stackwise_kernel_size,
    stackwise_stride,
    stackwise_se_ratio,
    stackwise_activation,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    alpha=1.0,
    **kwargs
)
```

Instantiates the MobileNetV3 architecture.

**References**

- [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
  (ICCV 2019)
  - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **stackwise_expansion**: list of ints or floats, the expansion ratio for
  each inverted residual block in the model.
- **stackwise_filters**: list of ints, number of filters for each inverted
  residual block in the model.
- **stackwise_stride**: list of ints, stride length for each inverted
  residual block in the model.
- **include_rescaling**: bool, whether to rescale the inputs. If set to True,
  inputs will be passed through a `Rescaling(scale=1 / 255)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e., output of `layers.Input()`)
  to use as image input for the model.
- **alpha**: float, controls the width of the network. This is known as the
  depth multiplier in the MobileNetV3 paper, but the name is kept for
  consistency with MobileNetV1 in Keras.
  - If `alpha` < 1.0, proportionally decreases the number
    of filters in each layer.
  - If `alpha` > 1.0, proportionally increases the number
    of filters in each layer.
  - If `alpha` = 1, default number of filters from the paper
    are used at each layer.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone with a custom config
model = MobileNetV3Backbone(
    stackwise_expansion=[1, 72.0 / 16, 88.0 / 24, 4, 6, 6, 3, 3, 6, 6, 6],
    stackwise_filters=[16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
    stackwise_kernel_size=[3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
    stackwise_stride=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
    stackwise_se_ratio=[0.25, None, None, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    stackwise_activation=["relu", "relu", "relu", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"],
    include_rescaling=False,
)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
MobileNetV3Backbone.from_preset()
```

Instantiate MobileNetV3Backbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.MobileNetV3Backbone.from_preset(
    "mobilenet_v3_large_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.MobileNetV3Backbone.from_preset(
    "mobilenet_v3_large_imagenet",
    load_weights=False,
```

| Preset name                 | Parameters | Description                                                                                                                                                                              |
| --------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| mobilenet_v3_small          | 933.50K    | MobileNetV3 model with 14 layers where the batch normalization and hard-swish activation are applied after the convolution layers.                                                       |
| mobilenet_v3_large          | 2.99M      | MobileNetV3 model with 28 layers where the batch normalization and hard-swish activation are applied after the convolution layers.                                                       |
| mobilenet_v3_large_imagenet | 2.99M      | MobileNetV3 model with 28 layers where the batch normalization and hard-swish activation are applied after the convolution layers. Pre-trained on the ImageNet 2012 classification task. |
| mobilenet_v3_small_imagenet | 933.50K    | MobileNetV3 model with 14 layers where the batch normalization and hard-swish activation are applied after the convolution layers. Pre-trained on the ImageNet 2012 classification task. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mobilenet_v3/mobilenet_v3_aliases.py#L54" >}}

### `MobileNetV3SmallBackbone` class

```python
keras_cv.models.MobileNetV3SmallBackbone(
    stackwise_expansion,
    stackwise_filters,
    stackwise_kernel_size,
    stackwise_stride,
    stackwise_se_ratio,
    stackwise_activation,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    alpha=1.0,
    **kwargs
)
```

MobileNetV3Backbone model with 14 layers.

**References**

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
  - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(scale=1 / 255)`
  layer. Defaults to True.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e., output of `layers.Input()`)
  to use as image input for the model.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = MobileNetV3SmallBackbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mobilenet_v3/mobilenet_v3_aliases.py#L88" >}}

### `MobileNetV3LargeBackbone` class

```python
keras_cv.models.MobileNetV3LargeBackbone(
    stackwise_expansion,
    stackwise_filters,
    stackwise_kernel_size,
    stackwise_stride,
    stackwise_se_ratio,
    stackwise_activation,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    alpha=1.0,
    **kwargs
)
```

MobileNetV3Backbone model with 28 layers.

**References**

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
  - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(scale=1 / 255)`
  layer. Defaults to True.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e., output of `layers.Input()`)
  to use as image input for the model.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = MobileNetV3LargeBackbone()
output = model(input_data)
```
