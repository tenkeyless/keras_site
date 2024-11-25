---
title: SegFormer Segmentation
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer.py#L31" >}}

### `SegFormer` class

```python
keras_cv.models.SegFormer(backbone, num_classes, projection_filters=256, **kwargs)
```

A Keras model implementing the SegFormer architecture for semantic
segmentation.

**References**

- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) # noqa: E501
- [Based on the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/tree/main/deepvision/models/segmentation/segformer) # noqa: E501

**Arguments**

- **backbone**: [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}). The backbone network for the model that is
  used as a feature extractor for the SegFormer encoder.
  It is _intended_ to be used only with the MiT backbone model which
  was created specifically for SegFormers. It should either be a
  `keras_cv.models.backbones.backbone.Backbone` or a [`tf.keras.Model`]({{< relref "/docs/api/models/model#model-class" >}})
  that implements the `pyramid_level_inputs` property with keys
  "P2", "P3", "P4", and "P5" and layer names as
  values.
- **num_classes**: int, the number of classes for the detection model,
  including the background class.
- **projection_filters**: int, number of filters in the
  convolution layer projecting the concatenated features into
  a segmentation map. Defaults to 256`.

**Example**

Using the class with a `backbone`:

```python
import tensorflow as tf
import keras_cv
images = np.ones(shape=(1, 96, 96, 3))
labels = np.zeros(shape=(1, 96, 96, 1))
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
model = keras_cv.models.segmentation.SegFormer(
    num_classes=1, backbone=backbone,
)
# Evaluate model
model(images)
# Train model
model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)
model.fit(images, labels, epochs=3)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer.py#L169" >}}

### `from_preset` method

```python
SegFormer.from_preset(
    preset, num_classes, load_weights=None, input_shape=None, **kwargs
)
```

Instantiate SegFormer model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5", "mit_b0_imagenet", "segformer_b0", "segformer_b1", "segformer_b2", "segformer_b3", "segformer_b4", "segformer_b5", "segformer_b0_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "segformer_b0_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.
- **input_shape** : input shape that will be passed to backbone
  initialization, Defaults to `None`.If `None`, the preset
  value will be used.

**Example**

```python
# Load architecture and weights from preset
model = keras_cv.models.SegFormer.from_preset(
    "segformer_b0_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.SegFormer.from_preset(
    "segformer_b0_imagenet",
    load_weights=False,
```

| Preset name           | Parameters | Description                                                                                                                           |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| mit_b0                | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks.                                                                                 |
| mit_b1                | 13.16M     | MiT (MixTransformer) model with 8 transformer blocks.                                                                                 |
| mit_b2                | 24.20M     | MiT (MixTransformer) model with 16 transformer blocks.                                                                                |
| mit_b3                | 44.08M     | MiT (MixTransformer) model with 28 transformer blocks.                                                                                |
| mit_b4                | 60.85M     | MiT (MixTransformer) model with 41 transformer blocks.                                                                                |
| mit_b5                | 81.45M     | MiT (MixTransformer) model with 52 transformer blocks.                                                                                |
| mit_b0_imagenet       | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks. Pre-trained on ImageNet-1K and scores 69% top-1 accuracy on the validation set. |
| segformer_b0          | 3.72M      | SegFormer model with MiTB0 backbone.                                                                                                  |
| segformer_b1          | 13.68M     | SegFormer model with MiTB1 backbone.                                                                                                  |
| segformer_b2          | 24.73M     | SegFormer model with MiTB2 backbone.                                                                                                  |
| segformer_b3          | 44.60M     | SegFormer model with MiTB3 backbone.                                                                                                  |
| segformer_b4          | 61.37M     | SegFormer model with MiTB4 backbone.                                                                                                  |
| segformer_b5          | 81.97M     | SegFormer model with MiTB5 backbone.                                                                                                  |
| segformer_b0_imagenet | 3.72M      | SegFormer model with a pretrained MiTB0 backbone.                                                                                     |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer_aliases.py#L43" >}}

### `SegFormerB0` class

```python
keras_cv.models.SegFormerB0(backbone, num_classes, projection_filters=256, **kwargs)
```

SegFormer model.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **backbone**: a KerasCV backbone for feature extraction.
- **num_classes**: the number of classes for segmentation, including the background class.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
segformer = keras_cv.models.SegFormer(backbone=backbone, num_classes=19)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer_aliases.py#L72" >}}

### `SegFormerB1` class

```python
keras_cv.models.SegFormerB1(backbone, num_classes, projection_filters=256, **kwargs)
```

SegFormer model.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **backbone**: a KerasCV backbone for feature extraction.
- **num_classes**: the number of classes for segmentation, including the background class.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
segformer = keras_cv.models.SegFormer(backbone=backbone, num_classes=19)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer_aliases.py#L101" >}}

### `SegFormerB2` class

```python
keras_cv.models.SegFormerB2(backbone, num_classes, projection_filters=256, **kwargs)
```

SegFormer model.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **backbone**: a KerasCV backbone for feature extraction.
- **num_classes**: the number of classes for segmentation, including the background class.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
segformer = keras_cv.models.SegFormer(backbone=backbone, num_classes=19)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer_aliases.py#L130" >}}

### `SegFormerB3` class

```python
keras_cv.models.SegFormerB3(backbone, num_classes, projection_filters=256, **kwargs)
```

SegFormer model.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **backbone**: a KerasCV backbone for feature extraction.
- **num_classes**: the number of classes for segmentation, including the background class.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
segformer = keras_cv.models.SegFormer(backbone=backbone, num_classes=19)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer_aliases.py#L159" >}}

### `SegFormerB4` class

```python
keras_cv.models.SegFormerB4(backbone, num_classes, projection_filters=256, **kwargs)
```

SegFormer model.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **backbone**: a KerasCV backbone for feature extraction.
- **num_classes**: the number of classes for segmentation, including the background class.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
segformer = keras_cv.models.SegFormer(backbone=backbone, num_classes=19)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/segformer/segformer_aliases.py#L188" >}}

### `SegFormerB5` class

```python
keras_cv.models.SegFormerB5(backbone, num_classes, projection_filters=256, **kwargs)
```

SegFormer model.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **backbone**: a KerasCV backbone for feature extraction.
- **num_classes**: the number of classes for segmentation, including the background class.

**Example**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
segformer = keras_cv.models.SegFormer(backbone=backbone, num_classes=19)
output = model(input_data)
```
