---
title: deeplab_v3_segmentation
toc: false
---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/segmentation/deeplab_v3_plus/deeplab_v3_plus.py#L33)

### `DeepLabV3Plus` class

`keras_cv.models.DeepLabV3Plus(     backbone,     num_classes,     projection_filters=48,     spatial_pyramid_pooling=None,     segmentation_head=None,     **kwargs )`

A Keras model implementing the DeepLabV3+ architecture for semantic segmentation.

**References**

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) # noqa: E501 (ECCV 2018) - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) # noqa: E501 (CVPR 2017)

**Arguments**

- **backbone**: [`keras.Model`](/api/models/model#model-class). The backbone network for the model that is used as a feature extractor for the DeepLabV3+ Encoder. Should either be a `keras_cv.models.backbones.backbone.Backbone` or a [`keras.Model`](/api/models/model#model-class) that implements the `pyramid_level_inputs` property with keys "P2" and "P5" and layer names as values. A somewhat sensible backbone to use in many cases is the `keras_cv.models.ResNet50V2Backbone.from_preset("resnet50_v2_imagenet")`.
- **num_classes**: int, the number of classes for the detection model. Note that the `num_classes` contains the background class, and the classes from the data should be represented by integers with range \[0, `num_classes`).
- **projection_filters**: int, number of filters in the convolution layer projecting low-level features from the `backbone`. The default value is set to `48`, as per the [TensorFlow implementation of DeepLab](https://github.com/tensorflow/models/blob/master/research/deeplab/model.py#L676). # noqa: E501
- **spatial_pyramid_pooling**: (Optional) a [`keras.layers.Layer`](/api/layers/base_layer#layer-class). Also known as Atrous Spatial Pyramid Pooling (ASPP). Performs spatial pooling on different spatial levels in the pyramid, with dilation. If provided, the feature map from the backbone is passed to it inside the DeepLabV3 Encoder, otherwise `keras_cv.layers.spatial_pyramid.SpatialPyramidPooling` is used.
- **segmentation_head**: (Optional) a [`keras.layers.Layer`](/api/layers/base_layer#layer-class). If provided, the outputs of the DeepLabV3 encoder is passed to this layer and it should predict the segmentation mask based on feature from backbone and feature from decoder, otherwise a default DeepLabV3 convolutional head is used.

**Example**

`import keras_cv  images = np.ones(shape=(1, 96, 96, 3)) labels = np.zeros(shape=(1, 96, 96, 1)) backbone = keras_cv.models.ResNet50V2Backbone(input_shape=[96, 96, 3]) model = keras_cv.models.segmentation.DeepLabV3Plus(     num_classes=1, backbone=backbone, )  # Evaluate model model(images)  # Train model model.compile(     optimizer="adam",     loss=keras.losses.BinaryCrossentropy(from_logits=False),     metrics=["accuracy"], ) model.fit(images, labels, epochs=3)`

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/task.py#L183)

### `from_preset` method

`DeepLabV3Plus.from_preset()`

Instantiate DeepLabV3Plus model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet18_v2", "resnet34_v2", "resnet50_v2", "resnet101_v2", "resnet152_v2", "mobilenet_v3_small", "mobilenet_v3_large", "csp_darknet_tiny", "csp_darknet_s", "csp_darknet_m", "csp_darknet_l", "csp_darknet_xl", "efficientnetv1_b0", "efficientnetv1_b1", "efficientnetv1_b2", "efficientnetv1_b3", "efficientnetv1_b4", "efficientnetv1_b5", "efficientnetv1_b6", "efficientnetv1_b7", "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_b0", "efficientnetv2_b1", "efficientnetv2_b2", "efficientnetv2_b3", "densenet121", "densenet169", "densenet201", "efficientnetlite_b0", "efficientnetlite_b1", "efficientnetlite_b2", "efficientnetlite_b3", "efficientnetlite_b4", "yolo_v8_xs_backbone", "yolo_v8_s_backbone", "yolo_v8_m_backbone", "yolo_v8_l_backbone", "yolo_v8_xl_backbone", "vitdet_base", "vitdet_large", "vitdet_huge", "videoswin_tiny", "videoswin_small", "videoswin_base", "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "deeplab_v3_plus_resnet50_pascalvoc". If looking for a preset with pretrained weights, choose one of "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "deeplab_v3_plus_resnet50_pascalvoc".
- **load_weights**: Whether to load pre-trained weights into model. Defaults to `None`, which follows whether the preset has pretrained weights available.
- **input_shape** : input shape that will be passed to backbone initialization, Defaults to `None`.If `None`, the preset value will be used.

**Example**

`# Load architecture and weights from preset model = keras_cv.models.DeepLabV3Plus.from_preset(     "resnet50_imagenet", )  # Load randomly initialized model from preset architecture with weights model = keras_cv.models.DeepLabV3Plus.from_preset(     "resnet50_imagenet",     load_weights=False,`

Preset name

Parameters

Description

deeplab_v3_plus_resnet50_pascalvoc

39.19M

DeeplabV3Plus with a ResNet50 v2 backbone. Trained on PascalVOC 2012 Semantic segmentation task, which consists of 20 classes and one background class. This model achieves a final categorical accuracy of 89.34% and mIoU of 0.6391 on evaluation dataset. This preset is only comptabile with Keras 3.

---
