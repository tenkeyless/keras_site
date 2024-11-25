---
title: ResNetV2 backbones
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v2/resnet_v2_backbone.py#L38" >}}

### `ResNetV2Backbone` class

```python
keras_cv.models.ResNetV2Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    stackwise_dilations=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

Instantiates the ResNetV2 architecture.

**Reference**

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

The difference in Resnet and ResNetV2 rests in the structure of their
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
- **stackwise_dilations**: list of ints, dilation for each stack in the
  model. If `None` (default), dilation will not be used.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **block_type**: string, one of "basic_block" or "block". The block type to
  stack. Use "basic_block" for smaller models like ResNet18 and
  ResNet34.

**Examples**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Pretrained backbone
model = keras_cv.models.ResNetV2Backbone.from_preset("resnet50_v2_imagenet")
output = model(input_data)
# Randomly initialized backbone with a custom config
model = ResNetV2Backbone(
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
ResNetV2Backbone.from_preset()
```

Instantiate ResNetV2Backbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "resnet18_v2", "resnet34_v2", "resnet50_v2", "resnet101_v2", "resnet152_v2", "resnet50_v2_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "resnet50_v2_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.ResNetV2Backbone.from_preset(
    "resnet50_v2_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.ResNetV2Backbone.from_preset(
    "resnet50_v2_imagenet",
    load_weights=False,
```

| Preset name          | Parameters | Description                                                                                                                                                            |
| -------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| resnet18_v2          | 11.18M     | ResNet model with 18 layers where the batch normalization and ReLU activation precede the convolution layers (v2 style).                                               |
| resnet34_v2          | 21.30M     | ResNet model with 34 layers where the batch normalization and ReLU activation precede the convolution layers (v2 style).                                               |
| resnet50_v2          | 23.56M     | ResNet model with 50 layers where the batch normalization and ReLU activation precede the convolution layers (v2 style).                                               |
| resnet101_v2         | 42.63M     | ResNet model with 101 layers where the batch normalization and ReLU activation precede the convolution layers (v2 style).                                              |
| resnet152_v2         | 58.33M     | ResNet model with 152 layers where the batch normalization and ReLU activation precede the convolution layers (v2 style).                                              |
| resnet50_v2_imagenet | 23.56M     | ResNet model with 50 layers where the batch normalization and ReLU activation precede the convolution layers (v2 style). Trained on Imagenet 2012 classification task. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v2/resnet_v2_aliases.py#L59" >}}

### `ResNet18V2Backbone` class

```python
keras_cv.models.ResNet18V2Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    stackwise_dilations=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetV2Backbone model with 18 layers.

**Reference**

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

The difference in ResNet and ResNetV2 rests in the structure of their
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
model = ResNet18V2Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v2/resnet_v2_aliases.py#L90" >}}

### `ResNet34V2Backbone` class

```python
keras_cv.models.ResNet34V2Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    stackwise_dilations=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetV2Backbone model with 34 layers.

**Reference**

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

The difference in ResNet and ResNetV2 rests in the structure of their
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
model = ResNet34V2Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v2/resnet_v2_aliases.py#L121" >}}

### `ResNet50V2Backbone` class

```python
keras_cv.models.ResNet50V2Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    stackwise_dilations=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetV2Backbone model with 50 layers.

**Reference**

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

The difference in ResNet and ResNetV2 rests in the structure of their
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
model = ResNet50V2Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v2/resnet_v2_aliases.py#L156" >}}

### `ResNet101V2Backbone` class

```python
keras_cv.models.ResNet101V2Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    stackwise_dilations=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetV2Backbone model with 101 layers.

**Reference**

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

The difference in ResNet and ResNetV2 rests in the structure of their
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
model = ResNet101V2Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/resnet_v2/resnet_v2_aliases.py#L187" >}}

### `ResNet152V2Backbone` class

```python
keras_cv.models.ResNet152V2Backbone(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    stackwise_dilations=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    block_type="block",
    **kwargs
)
```

ResNetV2Backbone model with 152 layers.

**Reference**

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

The difference in ResNet and ResNetV2 rests in the structure of their
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
model = ResNet152V2Backbone()
output = model(input_data)
```
