---
title: EfficientNetV2 B0 to B3 and S, M, L
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet_v2.py#L1086" >}}

### `EfficientNetV2B0` function

```python
keras.applications.EfficientNetV2B0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    name="efficientnetv2-b0",
)
```

Instantiates the EfficientNetV2B0 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded. Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is `False`.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to None.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `"max"` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000 (number of
  ImageNet classes).
- **classifier_activation**: A string or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `"softmax"`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet_v2.py#L1120" >}}

### `EfficientNetV2B1` function

```python
keras.applications.EfficientNetV2B1(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    name="efficientnetv2-b1",
)
```

Instantiates the EfficientNetV2B1 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded. Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is `False`.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to None.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `"max"` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000 (number of
  ImageNet classes).
- **classifier_activation**: A string or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `"softmax"`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet_v2.py#L1154" >}}

### `EfficientNetV2B2` function

```python
keras.applications.EfficientNetV2B2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    name="efficientnetv2-b2",
)
```

Instantiates the EfficientNetV2B2 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded. Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is `False`.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to None.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `"max"` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000 (number of
  ImageNet classes).
- **classifier_activation**: A string or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `"softmax"`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet_v2.py#L1188" >}}

### `EfficientNetV2B3` function

```python
keras.applications.EfficientNetV2B3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    name="efficientnetv2-b3",
)
```

Instantiates the EfficientNetV2B3 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded. Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is `False`.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to None.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `"max"` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000 (number of
  ImageNet classes).
- **classifier_activation**: A string or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `"softmax"`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet_v2.py#L1222" >}}

### `EfficientNetV2S` function

```python
keras.applications.EfficientNetV2S(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    name="efficientnetv2-s",
)
```

Instantiates the EfficientNetV2S architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded. Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is `False`.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to None.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `"max"` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000 (number of
  ImageNet classes).
- **classifier_activation**: A string or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `"softmax"`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet_v2.py#L1256" >}}

### `EfficientNetV2M` function

```python
keras.applications.EfficientNetV2M(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    name="efficientnetv2-m",
)
```

Instantiates the EfficientNetV2M architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded. Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is `False`.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to None.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `"max"` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000 (number of
  ImageNet classes).
- **classifier_activation**: A string or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `"softmax"`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/efficientnet_v2.py#L1290" >}}

### `EfficientNetV2L` function

```python
keras.applications.EfficientNetV2L(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    name="efficientnetv2-l",
)
```

Instantiates the EfficientNetV2L architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected
  layer at the top of the network. Defaults to `True`.
- **weights**: One of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded. Defaults to `"imagenet"`.
- **input_tensor**: Optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: Optional shape tuple, only to be specified
  if `include_top` is `False`.
  It should have exactly 3 inputs channels.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`. Defaults to None.
  - `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
  - `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
  - `"max"` means that global max pooling will
    be applied.
- **classes**: Optional number of classes to classify images
  into, only to be specified if `include_top` is `True`, and
  if no `weights` argument is specified. Defaults to 1000 (number of
  ImageNet classes).
- **classifier_activation**: A string or callable. The activation function to use
  on the "top" layer. Ignored unless `include_top=True`. Set
  `classifier_activation=None` to return the logits of the "top" layer.
  Defaults to `"softmax"`.
  When loading pretrained weights, `classifier_activation` can only
  be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.
