---
title: NasNetLarge and NasNetMobile
toc: true
weight: 9
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/nasnet.py#L411" >}}

### `NASNetLarge` function

```python
keras.applications.NASNetLarge(
    input_shape=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="nasnet_large",
)
```

Instantiates a NASNet model in ImageNet mode.

**Reference**

- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) (CVPR 2018)

Optionally loads weights pre-trained on ImageNet.
Note that the data format convention used by the model is
the one specified in your Keras config at `~/.keras/keras.json`.

Note: each Keras Application expects a specific kind of input preprocessing.
For NASNet, call `keras.applications.nasnet.preprocess_input` on your
inputs before passing them to the model.

**Arguments**

- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False (otherwise the input shape
  has to be `(331, 331, 3)` for NASNetLarge.
  It should have exactly 3 inputs channels,
  and width and height should be no smaller than 32.
  E.g. `(224, 224, 3)` would be one valid value.
- **include_top**: Whether to include the fully-connected
  layer at the top of the network.
- **weights**: `None` (random initialization) or
  `imagenet` (ImageNet weights). For loading `imagenet` weights,
  `input_shape` should be (331, 331, 3)
- **input_tensor**: Optional Keras tensor (i.e. output of
  `layers.Input()`)
  to use as image input for the model.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`.
  - `None` means that the output of the model
    will be the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a
    2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified.
- **classifier_activation**: A `str` or callable. The activation function to
  use on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top"
  layer. When loading pretrained weights, `classifier_activation`
  can only be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A Keras model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/nasnet.py#L315" >}}

### `NASNetMobile` function

```python
keras.applications.NASNetMobile(
    input_shape=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="nasnet_mobile",
)
```

Instantiates a Mobile NASNet model in ImageNet mode.

**Reference**

- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) (CVPR 2018)

Optionally loads weights pre-trained on ImageNet.
Note that the data format convention used by the model is
the one specified in your Keras config at `~/.keras/keras.json`.

Note: each Keras Application expects a specific kind of input preprocessing.
For NASNet, call `keras.applications.nasnet.preprocess_input` on your
inputs before passing them to the model.

**Arguments**

- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False (otherwise the input shape
  has to be `(224, 224, 3)` for NASNetMobile
  It should have exactly 3 inputs channels,
  and width and height should be no smaller than 32.
  E.g. `(224, 224, 3)` would be one valid value.
- **include_top**: Whether to include the fully-connected
  layer at the top of the network.
- **weights**: `None` (random initialization) or
  `imagenet` (ImageNet weights). For loading `imagenet` weights,
  `input_shape` should be (224, 224, 3)
- **input_tensor**: Optional Keras tensor (i.e. output of
  `layers.Input()`)
  to use as image input for the model.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`.
  - `None` means that the output of the model
    will be the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a
    2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified.
- **classifier_activation**: A `str` or callable. The activation function to
  use on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top"
  layer. When loading pretrained weights, `classifier_activation` can
  only be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A Keras model instance.
