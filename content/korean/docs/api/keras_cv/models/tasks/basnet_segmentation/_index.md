---
title: BASNet Segmentation
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/basnet/basnet.py#L37" >}}

### `BASNet` class

```python
keras_cv.models.BASNet(
    backbone,
    num_classes,
    input_shape=(None, None, 3),
    input_tensor=None,
    include_rescaling=False,
    projection_filters=64,
    prediction_heads=None,
    refinement_head=None,
    **kwargs
)
```

A Keras model implementing the BASNet architecture for semantic
segmentation.

**References**

- [BASNet: Boundary-Aware Segmentation Network for Mobile and Web Applications](https://arxiv.org/abs/2101.04704)

**Arguments**

- **backbone**: [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}). The backbone network for the model that is
  used as a feature extractor for BASNet prediction encoder. Currently
  supported backbones are ResNet18 and ResNet34. Default backbone is
  `keras_cv.models.ResNet34Backbone()`
  (Note: Do not specify 'input_shape', 'input_tensor', or 'include_rescaling'
  within the backbone. Please provide these while initializing the
  'BASNet' model.)
- **num_classes**: int, the number of classes for the segmentation model.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e., output of `layers.Input()`)
  to use as image input for the model.
- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **projection_filters**: int, number of filters in the convolution layer
  projecting low-level features from the `backbone`.
- **prediction_heads**: (Optional) List of [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) defining
  the prediction module head for the model. If not provided, a
  default head is created with a Conv2D layer followed by resizing.
- **refinement_head**: (Optional) a [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) defining the
  refinement module head for the model. If not provided, a default
  head is created with a Conv2D layer.

**Example**

```python
import keras_cv
images = np.ones(shape=(1, 288, 288, 3))
labels = np.zeros(shape=(1, 288, 288, 1))
# Note: Do not specify 'input_shape', 'input_tensor', or
# 'include_rescaling' within the backbone.
backbone = keras_cv.models.ResNet34Backbone()
model = keras_cv.models.segmentation.BASNet(
    backbone=backbone,
    num_classes=1,
    input_shape=[288, 288, 3],
    include_rescaling=False
)
# Evaluate model
output = model(images)
pred_labels = output[0]
# Train model
model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)
model.fit(images, labels, epochs=3)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/task.py#L183" >}}</span>

### `from_preset` method

```python
BASNet.from_preset()
```

Instantiate BASNet model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "resnet18", "resnet34", "basnet_resnet18", "basnet_resnet34".
  If looking for a preset with pretrained weights, choose one of
  "".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.
- **input_shape**: input shape that will be passed to backbone
  initialization, Defaults to `None`.If `None`, the preset
  value will be used.

**Example**

```python
model = keras_cv.models.BASNet.from_preset(
  "",
)

model = keras_cv.models.BASNet.from_preset(
  "",
  load_weights=False,
)
```

| Preset name     | Parameters | Description                         |
| --------------- | ---------- | ----------------------------------- |
| basnet_resnet18 | 98.78M     | BASNet with a ResNet18 v1 backbone. |
| basnet_resnet34 | 108.90M    | BASNet with a ResNet34 v1 backbone. |
