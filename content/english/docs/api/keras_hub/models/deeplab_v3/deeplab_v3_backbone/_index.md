---
title: DeepLabV3Backbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/deeplab_v3/deeplab_v3_backbone.py#L10" >}}

### `DeepLabV3Backbone` class

```python
keras_hub.models.DeepLabV3Backbone(
    image_encoder,
    spatial_pyramid_pooling_key,
    upsampling_size,
    dilation_rates,
    low_level_feature_key=None,
    projection_filters=48,
    image_shape=(None, None, 3),
    **kwargs
)
```

DeepLabV3 & DeepLabV3Plus architecture for semantic segmentation.

This class implements a DeepLabV3 & DeepLabV3Plus architecture as described
in [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)(ECCV 2018)
and [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)(CVPR 2017)

**Arguments**

- **image_encoder**: [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}). An instance that is used as a feature
  extractor for the Encoder. Should either be a
  [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) or a [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}) that implements the
  `pyramid_outputs` property with keys "P2", "P3" etc as values.
  A somewhat sensible backbone to use in many cases is
  the `keras_hub.models.ResNetBackbone.from_preset("resnet_v2_50")`.
- **projection_filters**: int. Number of filters in the convolution layer
  projecting low-level features from the `image_encoder`.
- **spatial_pyramid_pooling_key**: str. A layer level to extract and perform
  `spatial_pyramid_pooling`, one of the key from the `image_encoder`
  `pyramid_outputs` property such as "P4", "P5" etc.
- **upsampling_size**: int or tuple of 2 integers. The upsampling factors for
  rows and columns of `spatial_pyramid_pooling` layer.
  If `low_level_feature_key` is given then `spatial_pyramid_pooling`s
  layer resolution should match with the `low_level_feature`s layer
  resolution to concatenate both the layers for combined encoder
  outputs.
- **dilation_rates**: list. A `list` of integers for parallel dilated conv applied to
  `SpatialPyramidPooling`. Usually a
  sample choice of rates are `[6, 12, 18]`.
- **low_level_feature_key**: str optional. A layer level to extract the feature
  from one of the key from the `image_encoder`s `pyramid_outputs`
  property such as "P2", "P3" etc which will be the Decoder block.
  Required only when the DeepLabV3Plus architecture needs to be applied.
- **image_shape**: tuple. The input shape without the batch size.
  Defaults to `(None, None, 3)`.

**Example**

```python
# Load a trained backbone to extract features from it's `pyramid_outputs`.
image_encoder = keras_hub.models.ResNetBackbone.from_preset("resnet_50_imagenet")
model = keras_hub.models.DeepLabV3Backbone(
    image_encoder=image_encoder,
    projection_filters=48,
    low_level_feature_key="P2",
    spatial_pyramid_pooling_key="P5",
    upsampling_size = 8,
    dilation_rates = [6, 12, 18]
)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
DeepLabV3Backbone.from_preset(preset, load_weights=True, **kwargs)
```

Instantiate a [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as a
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

This constructor can be called in one of two ways. Either from the base
class like `keras_hub.models.Backbone.from_preset()`, or from
a model class like `keras_hub.models.GemmaBackbone.from_preset()`.
If calling from the base class, the subclass of the returning object
will be inferred from the config in the preset directory.

For any `Backbone` subclass, you can run `cls.presets.keys()` to list
all built-in presets available on the class.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the
  model architecture. If `False`, the weights will be randomly
  initialized.

**Examples**

```python
# Load a Gemma backbone with pre-trained weights.
model = keras_hub.models.Backbone.from_preset(
    "gemma_2b_en",
)
# Load a Bert backbone with a pre-trained config and random weights.
model = keras_hub.models.Backbone.from_preset(
    "bert_base_en",
    load_weights=False,
)
```

| Preset name                        | Parameters | Description                                                                                                                                                                                     |
| ---------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| deeplab_v3_plus_resnet50_pascalvoc | 39.19M     | DeepLabV3+ model with ResNet50 as image encoder and trained on augmented Pascal VOC dataset by Semantic Boundaries Dataset(SBD)which is having categorical accuracy of 90.01 and 0.63 Mean IoU. |
