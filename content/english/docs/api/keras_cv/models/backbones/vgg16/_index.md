---
title: VGG16 backbones
toc: true
weight: 9
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/vgg16/vgg16_backbone.py#L22" >}}

### `VGG16Backbone` class

```python
keras_cv.models.VGG16Backbone(
    include_rescaling,
    include_top,
    input_tensor=None,
    num_classes=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classifier_activation="softmax",
    name="VGG16",
    **kwargs
)
```

Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
  (ICLR 2015)
  This class represents Keras Backbone of VGG16 model.
  **Arguments**

* **include_rescaling**: bool, whether to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
* **include_top**: bool, whether to include the 3 fully-connected
  layers at the top of the network. If provided, num_classes must be
  provided.
* **num_classes**: int, optional number of classes to classify images into,
  only to be specified if `include_top` is True.
* **input_shape**: tuple, optional shape tuple, defaults to (224, 224, 3).
* **input_tensor**: Tensor, optional Keras tensor (i.e. output of
  `layers.Input()`) to use as image input for the model.
* **pooling**: bool, Optional pooling mode for feature extraction
  when `include_top` is `False`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional block.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional block, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
* **classifier_activation**:`str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
* **name**: (Optional) name to pass to the model, defaults to "VGG16".

**Returns**

A [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}) instance.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
VGG16Backbone.from_preset()
```

Not implemented.

No presets available for this class.
