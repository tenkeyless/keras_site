---
title: yolo_v8_detector
toc: false
---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/object_detection/yolo_v8/yolo_v8_detector.py#L323)

### `YOLOV8Detector` class

`keras_cv.models.YOLOV8Detector(     backbone,     num_classes,     bounding_box_format,     fpn_depth=2,     label_encoder=None,     prediction_decoder=None,     **kwargs )`

Implements the YOLOV8 architecture for object detection.

**Arguments**

- **backbone**: [`keras.Model`](/api/models/model#model-class), must implement the `pyramid_level_inputs` property with keys "P3", "P4", and "P5" and layer names as values. A sensible backbone to use is the [`keras_cv.models.YOLOV8Backbone`](/api/keras_cv/models/backbones/yolo_v8#yolov8backbone-class).
- **num_classes**: integer, the number of classes in your dataset excluding the background class. Classes should be represented by integers in the range \[0, num_classes).
- **bounding_box_format**: string, the format of bounding boxes of input dataset. Refer [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/) for more details on supported bounding box formats.
- **fpn_depth**: integer, a specification of the depth of the CSP blocks in the Feature Pyramid Network. This is usually 1, 2, or 3, depending on the size of your YOLOV8Detector model. We recommend using 3 for "yolo_v8_l_backbone" and "yolo_v8_xl_backbone". Defaults to 2.
- **label_encoder**: (Optional) A `YOLOV8LabelEncoder` that is responsible for transforming input boxes into trainable labels for YOLOV8Detector. If not provided, a default is provided.
- **prediction_decoder**: (Optional) A [`keras.layers.Layer`](/api/layers/base_layer#layer-class) that is responsible for transforming YOLOV8 predictions into usable bounding boxes. If not provided, a default is provided. The default `prediction_decoder` layer is a `keras_cv.layers.MultiClassNonMaxSuppression` layer, which uses a Non-Max Suppression for box pruning.

**Example**

`images = tf.ones(shape=(1, 512, 512, 3)) labels = {     "boxes": tf.constant([         [             [0, 0, 100, 100],             [100, 100, 200, 200],             [300, 300, 100, 100],         ]     ], dtype=tf.float32),     "classes": tf.constant([[1, 1, 1]], dtype=tf.int64), }  model = keras_cv.models.YOLOV8Detector(     num_classes=20,     bounding_box_format="xywh",     backbone=keras_cv.models.YOLOV8Backbone.from_preset(         "yolo_v8_m_backbone_coco"     ),     fpn_depth=2 )  # Evaluate model without box decoding and NMS model(images)  # Prediction with box decoding and NMS model.predict(images)  # Train model model.compile(     classification_loss='binary_crossentropy',     box_loss='ciou',     optimizer=tf.optimizers.SGD(global_clipnorm=10.0),     jit_compile=False, ) model.fit(images, labels)`

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/task.py#L183)

### `from_preset` method

`YOLOV8Detector.from_preset()`

Instantiate YOLOV8Detector model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet18_v2", "resnet34_v2", "resnet50_v2", "resnet101_v2", "resnet152_v2", "mobilenet_v3_small", "mobilenet_v3_large", "csp_darknet_tiny", "csp_darknet_s", "csp_darknet_m", "csp_darknet_l", "csp_darknet_xl", "efficientnetv1_b0", "efficientnetv1_b1", "efficientnetv1_b2", "efficientnetv1_b3", "efficientnetv1_b4", "efficientnetv1_b5", "efficientnetv1_b6", "efficientnetv1_b7", "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_b0", "efficientnetv2_b1", "efficientnetv2_b2", "efficientnetv2_b3", "densenet121", "densenet169", "densenet201", "efficientnetlite_b0", "efficientnetlite_b1", "efficientnetlite_b2", "efficientnetlite_b3", "efficientnetlite_b4", "yolo_v8_xs_backbone", "yolo_v8_s_backbone", "yolo_v8_m_backbone", "yolo_v8_l_backbone", "yolo_v8_xl_backbone", "vitdet_base", "vitdet_large", "vitdet_huge", "videoswin_tiny", "videoswin_small", "videoswin_base", "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "yolo_v8_m_pascalvoc". If looking for a preset with pretrained weights, choose one of "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "yolo_v8_m_pascalvoc".
- **load_weights**: Whether to load pre-trained weights into model. Defaults to `None`, which follows whether the preset has pretrained weights available.
- **input_shape** : input shape that will be passed to backbone initialization, Defaults to `None`.If `None`, the preset value will be used.

**Example**

`# Load architecture and weights from preset model = keras_cv.models.YOLOV8Detector.from_preset(     "resnet50_imagenet", )  # Load randomly initialized model from preset architecture with weights model = keras_cv.models.YOLOV8Detector.from_preset(     "resnet50_imagenet",     load_weights=False,`

Preset name

Parameters

Description

yolo_v8_m_pascalvoc

25.90M

YOLOV8-M pretrained on PascalVOC 2012 object detection task, which consists of 20 classes. This model achieves a final MaP of 0.45 on the evaluation set.

---
