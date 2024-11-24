---
title: image_classifier
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/classification/image_classifier.py#L31" >}}

### `ImageClassifier` class

`keras_cv.models.ImageClassifier(     backbone, num_classes, pooling="avg", activation="softmax", **kwargs )`

Image classifier with pooling and dense layer prediction head.

**Arguments**

- **backbone**: [`keras.Model`](/api/models/model#model-class) instance, the backbone architecture of the classifier called on the inputs. Pooling will be called on the last dimension of the backbone output.
- **num_classes**: int, number of classes to predict.
- **pooling**: str, type of pooling layer. Must be one of "avg", "max".
- **activation**: Optional `str` or callable, defaults to "softmax". The activation function to use on the Dense layer. Set `activation=None` to return the output logits.

**Example**

`input_data = tf.ones(shape=(8, 224, 224, 3))  # Pretrained classifier (e.g., for imagenet categories) model = keras_cv.models.ImageClassifier.from_preset(     "resnet50_v2_imagenet_classifier", ) output = model(input_data)  # Pretrained backbone backbone = keras_cv.models.ResNet50V2Backbone.from_preset(     "resnet50_v2_imagenet", ) model = keras_cv.models.ImageClassifier(     backbone=backbone,     num_classes=4, ) output = model(input_data)  # Randomly initialized backbone with a custom config model = keras_cv.models.ImageClassifier(     backbone=keras_cv.models.ResNet50V2Backbone(),     num_classes=4, ) output = model(input_data)`

---

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/task.py#L183" >}}

### `from_preset` method

`ImageClassifier.from_preset()`

Instantiate ImageClassifier model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet18_v2", "resnet34_v2", "resnet50_v2", "resnet101_v2", "resnet152_v2", "mobilenet_v3_small", "mobilenet_v3_large", "csp_darknet_tiny", "csp_darknet_s", "csp_darknet_m", "csp_darknet_l", "csp_darknet_xl", "efficientnetv1_b0", "efficientnetv1_b1", "efficientnetv1_b2", "efficientnetv1_b3", "efficientnetv1_b4", "efficientnetv1_b5", "efficientnetv1_b6", "efficientnetv1_b7", "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_b0", "efficientnetv2_b1", "efficientnetv2_b2", "efficientnetv2_b3", "densenet121", "densenet169", "densenet201", "efficientnetlite_b0", "efficientnetlite_b1", "efficientnetlite_b2", "efficientnetlite_b3", "efficientnetlite_b4", "yolo_v8_xs_backbone", "yolo_v8_s_backbone", "yolo_v8_m_backbone", "yolo_v8_l_backbone", "yolo_v8_xl_backbone", "vitdet_base", "vitdet_large", "vitdet_huge", "videoswin_tiny", "videoswin_small", "videoswin_base", "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "resnet50_v2_imagenet_classifier", "efficientnetv2_s_imagenet_classifier", "efficientnetv2_b0_imagenet_classifier", "efficientnetv2_b1_imagenet_classifier", "efficientnetv2_b2_imagenet_classifier", "mobilenet_v3_large_imagenet_classifier". If looking for a preset with pretrained weights, choose one of "resnet50_imagenet", "resnet50_v2_imagenet", "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet", "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet", "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet", "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b", "videoswin_tiny_kinetics400", "videoswin_small_kinetics400", "videoswin_base_kinetics400", "videoswin_base_kinetics400_imagenet22k", "videoswin_base_kinetics600_imagenet22k", "videoswin_base_something_something_v2", "resnet50_v2_imagenet_classifier", "efficientnetv2_s_imagenet_classifier", "efficientnetv2_b0_imagenet_classifier", "efficientnetv2_b1_imagenet_classifier", "efficientnetv2_b2_imagenet_classifier", "mobilenet_v3_large_imagenet_classifier".
- **load_weights**: Whether to load pre-trained weights into model. Defaults to `None`, which follows whether the preset has pretrained weights available.
- **input_shape** : input shape that will be passed to backbone initialization, Defaults to `None`.If `None`, the preset value will be used.

**Example**

`# Load architecture and weights from preset model = keras_cv.models.ImageClassifier.from_preset(     "resnet50_imagenet", )  # Load randomly initialized model from preset architecture with weights model = keras_cv.models.ImageClassifier.from_preset(     "resnet50_imagenet",     load_weights=False,`

Preset name

Parameters

Description

resnet50_v2_imagenet_classifier

25.61M

ResNet classifier with 50 layers where the batch normalization and ReLU activation precede the convolution layers (v2 style). Trained on Imagenet 2012 classification task.

efficientnetv2_s_imagenet_classifier

21.61M

ImageClassifier using the EfficientNet smallarchitecture. In this variant of the EfficientNet architecture, there are 6 convolutional blocks. Weights are initialized to pretrained imagenet classification weights.Published weights are capable of scoring 83.9% top 1 accuracy and 96.7% top 5 accuracy on imagenet.

efficientnetv2_b0_imagenet_classifier

7.20M

ImageClassifier using the EfficientNet B0 architecture. In this variant of the EfficientNet architecture, there are 6 convolutional blocks. As with all of the B style EfficientNet variants, the number of filters in each convolutional block is scaled by `width_coefficient=1.0` and `depth_coefficient=1.0`. Weights are initialized to pretrained imagenet classification weights. Published weights are capable of scoring 77.1% top 1 accuracy and 93.3% top 5 accuracy on imagenet.

efficientnetv2_b1_imagenet_classifier

8.21M

ImageClassifier using the EfficientNet B1 architecture. In this variant of the EfficientNet architecture, there are 6 convolutional blocks. As with all of the B style EfficientNet variants, the number of filters in each convolutional block is scaled by `width_coefficient=1.0` and `depth_coefficient=1.1`. Weights are initialized to pretrained imagenet classification weights.Published weights are capable of scoring 79.1% top 1 accuracy and 94.4% top 5 accuracy on imagenet.

efficientnetv2_b2_imagenet_classifier

10.18M

ImageClassifier using the EfficientNet B2 architecture. In this variant of the EfficientNet architecture, there are 6 convolutional blocks. As with all of the B style EfficientNet variants, the number of filters in each convolutional block is scaled by `width_coefficient=1.1` and `depth_coefficient1.2`. Weights are initialized to pretrained imagenet classification weights.Published weights are capable of scoring 80.1% top 1 accuracy and 94.9% top 5 accuracy on imagenet.

mobilenet_v3_large_imagenet_classifier

3.96M

ImageClassifier using the MobileNetV3Large architecture. This preset uses a Dense layer as a classification head instead of the typical fully-convolutional MobileNet head. As a result, it has fewer parameters than the original MobileNetV3Large model, which has 5.4 million parameters.Published weights are capable of scoring 69.4% top-1 accuracy and 89.4% top 5 accuracy on imagenet.

---
