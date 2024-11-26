---
title: YOLOV8 backbones
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/object_detection/yolo_v8/yolo_v8_backbone.py#L71" >}}

### `YOLOV8Backbone` class

```python
keras_cv.models.YOLOV8Backbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Implements the YOLOV8 backbone for object detection.

This backbone is a variant of the `CSPDarkNetBackbone` architecture.

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Arguments**

- **stackwise_channels**: A list of ints, the number of channels for each dark
  level in the model.
- **stackwise_depth**: A list of ints, the depth for each dark level in the
  model.
- **include_rescaling**: bool, whether to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **activation**: String. The activation functions to use in the backbone to
  use in the CSPDarkNet blocks. Defaults to "swish".
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

**Returns**

A [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}) instance.

**Examples**

```python
input_data = tf.ones(shape=(8, 224, 224, 3))
# Pretrained backbone
model = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xs_backbone_coco"
)
output = model(input_data)
# Randomly initialized backbone with a custom config
model = keras_cv.models.YOLOV8Backbone(
    stackwise_channels=[128, 256, 512, 1024],
    stackwise_depth=[3, 9, 9, 3],
    include_rescaling=False,
)
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
YOLOV8Backbone.from_preset()
```

Instantiate YOLOV8Backbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "yolo_v8_xs_backbone", "yolo_v8_s_backbone", "yolo_v8_m_backbone", "yolo_v8_l_backbone", "yolo_v8_xl_backbone", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco".
  If looking for a preset with pretrained weights, choose one of
  "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xs_backbone_coco",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xs_backbone_coco",
    load_weights=False,
```

| Preset name              | Parameters | Description                                       |
| ------------------------ | ---------- | ------------------------------------------------- |
| yolo_v8_xs_backbone      | 1.28M      | An extra small YOLOV8 backbone                    |
| yolo_v8_s_backbone       | 5.09M      | A small YOLOV8 backbone                           |
| yolo_v8_m_backbone       | 11.87M     | A medium YOLOV8 backbone                          |
| yolo_v8_l_backbone       | 19.83M     | A large YOLOV8 backbone                           |
| yolo_v8_xl_backbone      | 30.97M     | An extra large YOLOV8 backbone                    |
| yolo_v8_xs_backbone_coco | 1.28M      | An extra small YOLOV8 backbone pretrained on COCO |
| yolo_v8_s_backbone_coco  | 5.09M      | A small YOLOV8 backbone pretrained on COCO        |
| yolo_v8_m_backbone_coco  | 11.87M     | A medium YOLOV8 backbone pretrained on COCO       |
| yolo_v8_l_backbone_coco  | 19.83M     | A large YOLOV8 backbone pretrained on COCO        |
| yolo_v8_xl_backbone_coco | 30.97M     | An extra large YOLOV8 backbone pretrained on COCO |
