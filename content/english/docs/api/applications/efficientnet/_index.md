---
title: EfficientNet B0 to B7
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L555" >}}

### `EfficientNetB0` function

```python
keras.applications.EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb0",
)
```

Instantiates the EfficientNetB0 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L588" >}}

### `EfficientNetB1` function

```python
keras.applications.EfficientNetB1(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb1",
)
```

Instantiates the EfficientNetB1 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L621" >}}

### `EfficientNetB2` function

```python
keras.applications.EfficientNetB2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb2",
)
```

Instantiates the EfficientNetB2 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L654" >}}

### `EfficientNetB3` function

```python
keras.applications.EfficientNetB3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb3",
)
```

Instantiates the EfficientNetB3 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L687" >}}

### `EfficientNetB4` function

```python
keras.applications.EfficientNetB4(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb4",
)
```

Instantiates the EfficientNetB4 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L720" >}}

### `EfficientNetB5` function

```python
keras.applications.EfficientNetB5(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb5",
)
```

Instantiates the EfficientNetB5 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L753" >}}

### `EfficientNetB6` function

```python
keras.applications.EfficientNetB6(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb6",
)
```

Instantiates the EfficientNetB6 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet.py#L786" >}}

### `EfficientNetB7` function

```python
keras.applications.EfficientNetB7(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetb7",
)
```

Instantiates the EfficientNetB7 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

**Arguments**

- **include_top**: Whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
  Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is False.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to `None`.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `max` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is True, and
  if no `weights` argument is specified. 1000 is how many
  ImageNet classes there are. Defaults to `1000`.
- **classifier_activation**: A `str` or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `'softmax'`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.
