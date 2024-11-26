---
title: MixTransformer backbones
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mix_transformer/mix_transformer_backbone.py#L43" >}}

### `MiTBackbone` class

```python
keras_cv.models.MiTBackbone(
    include_rescaling,
    depths,
    input_shape=(224, 224, 3),
    input_tensor=None,
    embedding_dims=None,
    **kwargs
)
```

Base class for Backbone models.

Backbones are reusable layers of models trained on a standard task such as
Imagenet classification that can be reused in other tasks.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
MiTBackbone.from_preset()
```

Instantiate MiTBackbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5", "mit_b0_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "mit_b0_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.MiTBackbone.from_preset(
    "mit_b0_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.MiTBackbone.from_preset(
    "mit_b0_imagenet",
    load_weights=False,
```

| Preset name     | Parameters | Description                                                                                                                           |
| --------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| mit_b0          | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks.                                                                                 |
| mit_b1          | 13.16M     | MiT (MixTransformer) model with 8 transformer blocks.                                                                                 |
| mit_b2          | 24.20M     | MiT (MixTransformer) model with 16 transformer blocks.                                                                                |
| mit_b3          | 44.08M     | MiT (MixTransformer) model with 28 transformer blocks.                                                                                |
| mit_b4          | 60.85M     | MiT (MixTransformer) model with 41 transformer blocks.                                                                                |
| mit_b5          | 81.45M     | MiT (MixTransformer) model with 52 transformer blocks.                                                                                |
| mit_b0_imagenet | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks. Pre-trained on ImageNet-1K and scores 69% top-1 accuracy on the validation set. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mix_transformer/mix_transformer_aliases.py#L50" >}}

### `MiTB0Backbone` class

```python
keras_cv.models.MiTB0Backbone(
    include_rescaling,
    depths,
    input_shape=(224, 224, 3),
    input_tensor=None,
    embedding_dims=None,
    **kwargs
)
```

MiT model.

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
model = MiTB0Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mix_transformer/mix_transformer_aliases.py#L85" >}}

### `MiTB1Backbone` class

```python
keras_cv.models.MiTB1Backbone(
    include_rescaling,
    depths,
    input_shape=(224, 224, 3),
    input_tensor=None,
    embedding_dims=None,
    **kwargs
)
```

MiT model.

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
model = MiTB1Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mix_transformer/mix_transformer_aliases.py#L115" >}}

### `MiTB2Backbone` class

```python
keras_cv.models.MiTB2Backbone(
    include_rescaling,
    depths,
    input_shape=(224, 224, 3),
    input_tensor=None,
    embedding_dims=None,
    **kwargs
)
```

MiT model.

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
model = MiTB2Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mix_transformer/mix_transformer_aliases.py#L145" >}}

### `MiTB3Backbone` class

```python
keras_cv.models.MiTB3Backbone(
    include_rescaling,
    depths,
    input_shape=(224, 224, 3),
    input_tensor=None,
    embedding_dims=None,
    **kwargs
)
```

MiT model.

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
model = MiTB3Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mix_transformer/mix_transformer_aliases.py#L175" >}}

### `MiTB4Backbone` class

```python
keras_cv.models.MiTB4Backbone(
    include_rescaling,
    depths,
    input_shape=(224, 224, 3),
    input_tensor=None,
    embedding_dims=None,
    **kwargs
)
```

MiT model.

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
model = MiTB4Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/mix_transformer/mix_transformer_aliases.py#L205" >}}

### `MiTB5Backbone` class

```python
keras_cv.models.MiTB5Backbone(
    include_rescaling,
    depths,
    input_shape=(224, 224, 3),
    input_tensor=None,
    embedding_dims=None,
    **kwargs
)
```

MiT model.

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
model = MiTB5Backbone()
output = model(input_data)
```
