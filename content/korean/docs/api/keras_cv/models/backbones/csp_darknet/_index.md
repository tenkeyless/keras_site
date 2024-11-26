---
title: CSPDarkNet backbones
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/csp_darknet/csp_darknet_backbone.py#L44" >}}

### `CSPDarkNetBackbone` class

```python
keras_cv.models.CSPDarkNetBackbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    use_depthwise=False,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

This class represents the CSPDarkNet architecture.

**Reference**

- [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
  - [CSPNet Paper](https://arxiv.org/abs/1911.11929)
  - [YoloX Paper](https://arxiv.org/abs/2107.08430)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **stackwise_channels**: A list of ints, the number of channels for each dark
  level in the model.
- **stackwise_depth**: A list of ints, the depth for each dark level in the
  model.
- **include_rescaling**: bool, whether to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **use_depthwise**: bool, whether a `DarknetConvBlockDepthwise` should be
  used over a `DarknetConvBlock`, defaults to False.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of
  `keras.layers.Input()`) to use as image input for the model.

**Returns**

A [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}) instance.

**Examples**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Pretrained backbone
model = keras_cv.models.CSPDarkNetBackbone.from_preset(
    "csp_darknet_tiny_imagenet"
)
output = model(input_data)
# Randomly initialized backbone with a custom config
model = keras_cv.models.CSPDarkNetBackbone(
    stackwise_channels=[128, 256, 512, 1024],
    stackwise_depth=[3, 9, 9, 3],
    include_rescaling=False,
)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
CSPDarkNetBackbone.from_preset()
```

Instantiate CSPDarkNetBackbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "csp_darknet_tiny", "csp_darknet_s", "csp_darknet_m", "csp_darknet_l", "csp_darknet_xl", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.CSPDarkNetBackbone.from_preset(
    "csp_darknet_tiny_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.CSPDarkNetBackbone.from_preset(
    "csp_darknet_tiny_imagenet",
    load_weights=False,
```

| Preset name               | Parameters | Description                                                                                                                                                                                                            |
| ------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| csp_darknet_tiny          | 2.38M      | CSPDarkNet model with [48, 96, 192, 384] channels and [1, 3, 3, 1] depths where the batch normalization and SiLU activation are applied after the convolution layers.                                                  |
| csp_darknet_s             | 4.22M      | CSPDarkNet model with [64, 128, 256, 512] channels and [1, 3, 3, 1] depths where the batch normalization and SiLU activation are applied after the convolution layers.                                                 |
| csp_darknet_m             | 12.37M     | CSPDarkNet model with [96, 192, 384, 768] channels and [2, 6, 6, 2] depths where the batch normalization and SiLU activation are applied after the convolution layers.                                                 |
| csp_darknet_l             | 27.11M     | CSPDarkNet model with [128, 256, 512, 1024] channels and [3, 9, 9, 3] depths where the batch normalization and SiLU activation are applied after the convolution layers.                                               |
| csp_darknet_xl            | 56.84M     | CSPDarkNet model with [170, 340, 680, 1360] channels and [4, 12, 12, 4] depths where the batch normalization and SiLU activation are applied after the convolution layers.                                             |
| csp_darknet_tiny_imagenet | 2.38M      | CSPDarkNet model with [48, 96, 192, 384] channels and [1, 3, 3, 1] depths where the batch normalization and SiLU activation are applied after the convolution layers. Trained on Imagenet 2012 classification task.    |
| csp_darknet_l_imagenet    | 27.11M     | CSPDarkNet model with [128, 256, 512, 1024] channels and [3, 9, 9, 3] depths where the batch normalization and SiLU activation are applied after the convolution layers. Trained on Imagenet 2012 classification task. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/csp_darknet/csp_darknet_aliases.py#L55" >}}

### `CSPDarkNetTinyBackbone` class

```python
keras_cv.models.CSPDarkNetTinyBackbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    use_depthwise=False,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

CSPDarkNetBackbone model with [48, 96, 192, 384] channels
and [1, 3, 3, 1] depths.

**Reference**

- [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
  - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
  - [YoloX Paper](https://arxiv.org/abs/2107.08430)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether or not to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = CSPDarkNetTinyBackbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/csp_darknet/csp_darknet_aliases.py#L90" >}}

### `CSPDarkNetSBackbone` class

```python
keras_cv.models.CSPDarkNetSBackbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    use_depthwise=False,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

CSPDarkNetBackbone model with [64, 128, 256, 512] channels
and [1, 3, 3, 1] depths.

**Reference**

- [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
  - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
  - [YoloX Paper](https://arxiv.org/abs/2107.08430)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether or not to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = CSPDarkNetSBackbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/csp_darknet/csp_darknet_aliases.py#L121" >}}

### `CSPDarkNetMBackbone` class

```python
keras_cv.models.CSPDarkNetMBackbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    use_depthwise=False,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

CSPDarkNetBackbone model with [96, 192, 384, 768] channels
and [2, 6, 6, 2] depths.

**Reference**

- [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
  - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
  - [YoloX Paper](https://arxiv.org/abs/2107.08430)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether or not to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = CSPDarkNetMBackbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/csp_darknet/csp_darknet_aliases.py#L152" >}}

### `CSPDarkNetLBackbone` class

```python
keras_cv.models.CSPDarkNetLBackbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    use_depthwise=False,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

CSPDarkNetBackbone model with [128, 256, 512, 1024] channels
and [3, 9, 9, 3] depths.

**Reference**

- [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
  - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
  - [YoloX Paper](https://arxiv.org/abs/2107.08430)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether or not to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = CSPDarkNetLBackbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/csp_darknet/csp_darknet_aliases.py#L187" >}}

### `CSPDarkNetXLBackbone` class

```python
keras_cv.models.CSPDarkNetXLBackbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    use_depthwise=False,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

CSPDarkNetBackbone model with [170, 340, 680, 1360] channels
and [4, 12, 12, 4] depths.

**Reference**

- [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
  - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
  - [YoloX Paper](https://arxiv.org/abs/2107.08430)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **include_rescaling**: bool, whether or not to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = CSPDarkNetXLBackbone()
output = model(input_data)
```
