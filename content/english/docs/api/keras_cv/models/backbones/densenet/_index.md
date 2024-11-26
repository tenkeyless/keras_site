---
title: DenseNet backbones
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/densenet/densenet_backbone.py#L40" >}}

### `DenseNetBackbone` class

```python
keras_cv.models.DenseNetBackbone(
    stackwise_num_repeats,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    compression_ratio=0.5,
    growth_rate=32,
    **kwargs
)
```

Instantiates the DenseNet architecture.

**Arguments**

- **stackwise_num_repeats**: list of ints, number of repeated convolutional
  blocks per dense block.
- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of
  `keras.layers.Input()`) to use as image input for the model.
- **compression_ratio**: float, compression rate at transition layers.
- **growth_rate**: int, number of filters added by each dense block.

**Examples**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Pretrained backbone
model = keras_cv.models.DenseNetBackbone.from_preset("densenet121_imagenet")
output = model(input_data)
# Randomly initialized backbone with a custom config
model = DenseNetBackbone(
    stackwise_num_repeats=[6, 12, 24, 16],
    include_rescaling=False,
)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
DenseNetBackbone.from_preset()
```

Instantiate DenseNetBackbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "densenet121", "densenet169", "densenet201", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.DenseNetBackbone.from_preset(
    "densenet121_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.DenseNetBackbone.from_preset(
    "densenet121_imagenet",
    load_weights=False,
```

| Preset name          | Parameters | Description                                                                   |
| -------------------- | ---------- | ----------------------------------------------------------------------------- |
| densenet121          | Unknown    | DenseNet model with 121 layers.                                               |
| densenet169          | Unknown    | DenseNet model with 169 layers.                                               |
| densenet201          | Unknown    | DenseNet model with 201 layers.                                               |
| densenet121_imagenet | Unknown    | DenseNet model with 121 layers. Trained on Imagenet 2012 classification task. |
| densenet169_imagenet | Unknown    | DenseNet model with 169 layers. Trained on Imagenet 2012 classification task. |
| densenet201_imagenet | Unknown    | DenseNet model with 201 layers. Trained on Imagenet 2012 classification task. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/densenet/densenet_aliases.py#L53" >}}

### `DenseNet121Backbone` class

```python
keras_cv.models.DenseNet121Backbone(
    stackwise_num_repeats,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    compression_ratio=0.5,
    growth_rate=32,
    **kwargs
)
```

DenseNetBackbone model with 121 layers.

**Reference**

- [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

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
model = DenseNet121Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/densenet/densenet_aliases.py#L87" >}}

### `DenseNet169Backbone` class

```python
keras_cv.models.DenseNet169Backbone(
    stackwise_num_repeats,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    compression_ratio=0.5,
    growth_rate=32,
    **kwargs
)
```

DenseNetBackbone model with 169 layers.

**Reference**

- [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

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
model = DenseNet169Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/densenet/densenet_aliases.py#L121" >}}

### `DenseNet201Backbone` class

```python
keras_cv.models.DenseNet201Backbone(
    stackwise_num_repeats,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    compression_ratio=0.5,
    growth_rate=32,
    **kwargs
)
```

DenseNetBackbone model with 201 layers.

**Reference**

- [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

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
model = DenseNet201Backbone()
output = model(input_data)
```
