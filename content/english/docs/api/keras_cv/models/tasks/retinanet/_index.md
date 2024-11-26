---
title: The RetinaNet model
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/object_detection/retinanet/retinanet.py#L44" >}}

### `RetinaNet` class

```python
keras_cv.models.RetinaNet(
    backbone,
    num_classes,
    bounding_box_format,
    anchor_generator=None,
    label_encoder=None,
    prediction_decoder=None,
    feature_pyramid=None,
    classification_head=None,
    box_head=None,
    **kwargs
)
```

A Keras model implementing the RetinaNet meta-architecture.

Implements the RetinaNet architecture for object detection. The constructor
requires `num_classes`, `bounding_box_format`, and a backbone. Optionally,
a custom label encoder, and prediction decoder may be provided.

**Example**

```python
images = np.ones((1, 512, 512, 3))
labels = {
    "boxes": tf.cast([
        [
            [0, 0, 100, 100],
            [100, 100, 200, 200],
            [300, 300, 100, 100],
        ]
    ], dtype=tf.float32),
    "classes": tf.cast([[1, 1, 1]], dtype=tf.float32),
}
model = keras_cv.models.RetinaNet(
    num_classes=20,
    bounding_box_format="xywh",
    backbone=keras_cv.models.ResNet50Backbone.from_preset(
        "resnet50_imagenet"
    )
)
# Evaluate model without box decoding and NMS
model(images)
# Prediction with box decoding and NMS
model.predict(images)
# Train model
model.compile(
    classification_loss='focal',
    box_loss='smoothl1',
    optimizer=keras.optimizers.SGD(global_clipnorm=10.0),
    jit_compile=False,
)
model.fit(images, labels)
```

**Arguments**

- **num_classes**: the number of classes in your dataset excluding the
  background class. Classes should be represented by integers in the
  range [0, num_classes).
- **bounding_box_format**: The format of bounding boxes of input dataset.
  Refer
  [to the keras.io docs]({{< relref "/docs/api/keras_cv/bounding_box/formats/" >}})
  for more details on supported bounding box formats.
- **backbone**: [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}). If the default `feature_pyramid` is used,
  must implement the `pyramid_level_inputs` property with keys "P3", "P4",
  and "P5" and layer names as values. A somewhat sensible backbone
  to use in many cases is the:
  `keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")`
- **anchor_generator**: (Optional) a `keras_cv.layers.AnchorGenerator`. If
  provided, the anchor generator will be passed to both the
  `label_encoder` and the `prediction_decoder`. Only to be used when
  both `label_encoder` and `prediction_decoder` are both `None`.
  Defaults to an anchor generator with the parameterization:
  `strides=[2**i for i in range(3, 8)]`,
  `scales=[2**x for x in [0, 1 / 3, 2 / 3]]`,
  `sizes=[32.0, 64.0, 128.0, 256.0, 512.0]`,
  and `aspect_ratios=[0.5, 1.0, 2.0]`.
- **label_encoder**: (Optional) a keras.Layer that accepts an image Tensor, a
  bounding box Tensor and a bounding box class Tensor to its `call()`
  method, and returns RetinaNet training targets. By default, a
  KerasCV standard `RetinaNetLabelEncoder` is created and used.
  Results of this object's `call()` method are passed to the `loss`
  object for `box_loss` and `classification_loss` the `y_true`
  argument.
- **prediction_decoder**: (Optional) A [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) that is
  responsible for transforming RetinaNet predictions into usable
  bounding box Tensors. If not provided, a default is provided. The
  default `prediction_decoder` layer is a
  `keras_cv.layers.MultiClassNonMaxSuppression` layer, which uses
  a Non-Max Suppression for box pruning.
- **feature_pyramid**: (Optional) A [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) that produces
  a list of 4D feature maps (batch dimension included)
  when called on the pyramid-level outputs of the `backbone`.
  If not provided, the reference implementation from the paper will be used.
- **classification_head**: (Optional) A `keras.Layer` that performs
  classification of the bounding boxes. If not provided, a simple
  ConvNet with 3 layers will be used.
- **box_head**: (Optional) A `keras.Layer` that performs regression of the
  bounding boxes. If not provided, a simple ConvNet with 3 layers
  will be used.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/task.py#L183" >}}

### `from_preset` method

```python
RetinaNet.from_preset()
```

Instantiate RetinaNet model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet18_v2", "resnet34_v2", "resnet50_v2", "resnet101_v2", "resnet152_v2", "mobilenet_v3_small", "mobilenet_v3_large", "csp_darknet_tiny", "csp_darknet_s", "csp_darknet_m", "csp_darknet_l", "csp_darknet_xl", "efficientnetv1_b0", "efficientnetv1_b1", "efficientnetv1_b2", "efficientnetv1_b3", "efficientnetv1_b4", "efficientnetv1_b5", "efficientnetv1_b6", "efficientnetv1_b7", "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_b0", "efficientnetv2_b1", "efficientnetv2_b2", "efficientnetv2_b3", "densenet121", "densenet169", "densenet201", "efficientnetlite_b0", "efficientnetlite_b1", "efficientnetlite_b2", "efficientnetlite_b3", "efficientnetlite_b4", "yolo_v8_xs_backbone", "yolo_v8_s_backbone", "yolo_v8_m_backbone", "yolo_v8_l_backbone", "yolo_v8_xl_backbone", "vitdet_base", "vitdet_large", "vitdet_huge", "videoswin_tiny", "videoswin_small", "videoswin_base", "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "retinanet_resnet50_pascalvoc".
  If looking for a preset with pretrained weights, choose one of
  "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "retinanet_resnet50_pascalvoc".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.
- **input_shape** : input shape that will be passed to backbone
  initialization, Defaults to `None`.If `None`, the preset
  value will be used.

**Example**

```python
# Load architecture and weights from preset
model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    load_weights=False,
```

| Preset name                  | Parameters | Description                                                                                                                                                                          |
| ---------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| retinanet_resnet50_pascalvoc | 35.60M     | RetinaNet with a ResNet50 v1 backbone. Trained on PascalVOC 2012 object detection task, which consists of 20 classes. This model achieves a final MaP of 0.33 on the evaluation set. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/object_detection/retinanet/prediction_head.py#L19" >}}

### `PredictionHead` class

```python
keras_cv.models.retinanet.PredictionHead(
    output_filters, bias_initializer, num_conv_layers=3, **kwargs
)
```

The class/box predictions head.

**Arguments**

- **output_filters**: Number of convolution filters in the final layer.
- **bias_initializer**: Bias Initializer for the final convolution layer.

**Returns**

A function representing either the classification
or the box regression head depending on `output_filters`.
