---
title: ResNetV1 backbones
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v1/resnet_v1_backbone.py#L39" >}}

### `ResNetBackbone` class

```python
keras_cv.models.ResNetBackbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

Instantiates the ResNet architecture.

**Reference**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The difference in ResNetV1 and ResNetV2 rests in the structure of their
individual building blocks. In ResNetV2, the batch normalization and
ReLU activation precede the convolution layers, as opposed to ResNetV1 where
the batch normalization and ReLU activation are applied after the
convolution layers.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **stackwise_filters**: list of ints, number of filters for each stack in
  the model.
- **stackwise_blocks**: list of ints, number of blocks for each stack in the
  model.
- **stackwise_strides**: list of ints, stride for each stack in the model.
- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **block_type**: string, one of "basic_block" or "block". The block type to
  stack. Use "basic_block" for ResNet18 and ResNet34.

**Examples**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Pretrained backbone
model = keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")
output = model(input_data)
# Randomly initialized backbone with a custom config
model = ResNetBackbone(
    stackwise_filters=[64, 128, 256, 512],
    stackwise_blocks=[2, 2, 2, 2],
    stackwise_strides=[1, 2, 2, 2],
    include_rescaling=False,
)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
ResNetBackbone.from_preset()
```

Instantiate ResNetBackbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet50_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "resnet50_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.ResNetBackbone.from_preset(
    "resnet50_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.ResNetBackbone.from_preset(
    "resnet50_imagenet",
    load_weights=False,
```

| Preset name       | Parameters | Description                                                                                                                                                                      |
| ----------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| resnet18          | 11.19M     | ResNet model with 18 layers where the batch normalization and ReLU activation are applied after the convolution layers (v1 style).                                               |
| resnet34          | 21.30M     | ResNet model with 34 layers where the batch normalization and ReLU activation are applied after the convolution layers (v1 style).                                               |
| resnet50          | 23.56M     | ResNet model with 50 layers where the batch normalization and ReLU activation are applied after the convolution layers (v1 style).                                               |
| resnet101         | 42.61M     | ResNet model with 101 layers where the batch normalization and ReLU activation are applied after the convolution layers (v1 style).                                              |
| resnet152         | 58.30M     | ResNet model with 152 layers where the batch normalization and ReLU activation are applied after the convolution layers (v1 style).                                              |
| resnet50_imagenet | 23.56M     | ResNet model with 50 layers where the batch normalization and ReLU activation are applied after the convolution layers (v1 style). Trained on Imagenet 2012 classification task. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v1/resnet_v1_aliases.py#L59" >}}

### `ResNet18Backbone` class

```python
keras_cv.models.ResNet18Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetBackbone (V1) model with 18 layers.

**Reference**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The difference in ResNetV1 and ResNetV2 rests in the structure of their
individual building blocks. In ResNetV2, the batch normalization and
ReLU activation precede the convolution layers, as opposed to ResNetV1 where
the batch normalization and ReLU activation are applied after the
convolution layers.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = ResNet18Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v1/resnet_v1_aliases.py#L90" >}}

### `ResNet34Backbone` class

```python
keras_cv.models.ResNet34Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetBackbone (V1) model with 34 layers.

**Reference**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The difference in ResNetV1 and ResNetV2 rests in the structure of their
individual building blocks. In ResNetV2, the batch normalization and
ReLU activation precede the convolution layers, as opposed to ResNetV1 where
the batch normalization and ReLU activation are applied after the
convolution layers.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = ResNet34Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v1/resnet_v1_aliases.py#L121" >}}

### `ResNet50Backbone` class

```python
keras_cv.models.ResNet50Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetBackbone (V1) model with 50 layers.

**Reference**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The difference in ResNetV1 and ResNetV2 rests in the structure of their
individual building blocks. In ResNetV2, the batch normalization and
ReLU activation precede the convolution layers, as opposed to ResNetV1 where
the batch normalization and ReLU activation are applied after the
convolution layers.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = ResNet50Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v1/resnet_v1_aliases.py#L156" >}}

### `ResNet101Backbone` class

```python
keras_cv.models.ResNet101Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetBackbone (V1) model with 101 layers.

**Reference**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The difference in ResNetV1 and ResNetV2 rests in the structure of their
individual building blocks. In ResNetV2, the batch normalization and
ReLU activation precede the convolution layers, as opposed to ResNetV1 where
the batch normalization and ReLU activation are applied after the
convolution layers.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = ResNet101Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v1/resnet_v1_aliases.py#L187" >}}

### `ResNet152Backbone` class

```python
keras_cv.models.ResNet152Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetBackbone (V1) model with 152 layers.

**Reference**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The difference in ResNetV1 and ResNetV2 rests in the structure of their
individual building blocks. In ResNetV2, the batch normalization and
ReLU activation precede the convolution layers, as opposed to ResNetV1 where
the batch normalization and ReLU activation are applied after the
convolution layers.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = ResNet152Backbone()
output = model(input_data)
```
