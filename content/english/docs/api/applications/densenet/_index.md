---
title: DenseNet
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/densenet.py#L321" >}}

### `DenseNet121` function

```python
keras.applications.DenseNet121(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="densenet121",
)
```

Instantiates the Densenet121 architecture.

**Reference**

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

Optionally loads weights pre-trained on ImageNet.
Note that the data format convention used by the model is
the one specified in your Keras config at `~/.keras/keras.json`.

Note: each Keras Application expects a specific kind of input preprocessing.
For DenseNet, call `keras.applications.densenet.preprocess_input`
on your inputs before passing them to the model.

**Arguments**

- **include_top**: whether to include the fully-connected
  layer at the top of the network.
- **weights**: one of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  input_tensor: optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, only to be specified
  if `include_top` is False (otherwise the input shape
  has to be `(224, 224, 3)` (with `'channels_last'` data format)
  or `(3, 224, 224)` (with `'channels_first'` data format).
  It should have exactly 3 inputs channels,
  and width and height should be no smaller than 32.
  E.g. `(200, 200, 3)` would be one valid value.
- **pooling**: Optional pooling mode for feature extraction
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
- **classes**: optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000.
- **classifier_activation**: A `str` or callable.
  The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits
  of the "top" layer. When loading pretrained weights,
  `classifier_activation` can only be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A Keras model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/densenet.py#L351" >}}

### `DenseNet169` function

```python
keras.applications.DenseNet169(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="densenet169",
)
```

Instantiates the Densenet169 architecture.

**Reference**

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

Optionally loads weights pre-trained on ImageNet.
Note that the data format convention used by the model is
the one specified in your Keras config at `~/.keras/keras.json`.

Note: each Keras Application expects a specific kind of input preprocessing.
For DenseNet, call `keras.applications.densenet.preprocess_input`
on your inputs before passing them to the model.

**Arguments**

- **include_top**: whether to include the fully-connected
  layer at the top of the network.
- **weights**: one of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  input_tensor: optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, only to be specified
  if `include_top` is False (otherwise the input shape
  has to be `(224, 224, 3)` (with `'channels_last'` data format)
  or `(3, 224, 224)` (with `'channels_first'` data format).
  It should have exactly 3 inputs channels,
  and width and height should be no smaller than 32.
  E.g. `(200, 200, 3)` would be one valid value.
- **pooling**: Optional pooling mode for feature extraction
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
- **classes**: optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000.
- **classifier_activation**: A `str` or callable.
  The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits
  of the "top" layer. When loading pretrained weights,
  `classifier_activation` can only be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A Keras model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/densenet.py#L381" >}}

### `DenseNet201` function

```python
keras.applications.DenseNet201(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="densenet201",
)
```

Instantiates the Densenet201 architecture.

**Reference**

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

Optionally loads weights pre-trained on ImageNet.
Note that the data format convention used by the model is
the one specified in your Keras config at `~/.keras/keras.json`.

Note: each Keras Application expects a specific kind of input preprocessing.
For DenseNet, call `keras.applications.densenet.preprocess_input`
on your inputs before passing them to the model.

**Arguments**

- **include_top**: whether to include the fully-connected
  layer at the top of the network.
- **weights**: one of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  input_tensor: optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, only to be specified
  if `include_top` is False (otherwise the input shape
  has to be `(224, 224, 3)` (with `'channels_last'` data format)
  or `(3, 224, 224)` (with `'channels_first'` data format).
  It should have exactly 3 inputs channels,
  and width and height should be no smaller than 32.
  E.g. `(200, 200, 3)` would be one valid value.
- **pooling**: Optional pooling mode for feature extraction
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
- **classes**: optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000.
- **classifier_activation**: A `str` or callable.
  The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits
  of the "top" layer. When loading pretrained weights,
  `classifier_activation` can only be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A Keras model instance.
