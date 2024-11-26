---
title: InceptionResNetV2
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/applications/inception_resnet_v2.py#L16" >}}

### `InceptionResNetV2` function

```python
keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="inception_resnet_v2",
)
```

Instantiates the Inception-ResNet v2 architecture.

**Reference**

- [Inception-v4, Inception-ResNet and the Impact of
  Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
  (AAAI 2017)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples]({{< relref "/docs/api/applications/#usage-examples-for-image-classification-models" >}}).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

Note: each Keras Application expects a specific kind of
input preprocessing. For InceptionResNetV2, call
`keras.applications.inception_resnet_v2.preprocess_input`
on your inputs before passing them to the model.
`inception_resnet_v2.preprocess_input`
will scale input pixels between -1 and 1.

**Arguments**

- **include_top**: whether to include the fully-connected
  layer at the top of the network.
- **weights**: one of `None` (random initialization),
  `"imagenet"` (pre-training on ImageNet),
  or the path to the weights file to be loaded.
- **input_tensor**: optional Keras tensor
  (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **input_shape**: optional shape tuple, only to be specified
  if `include_top` is `False` (otherwise the input shape
  has to be `(299, 299, 3)`
  (with `'channels_last'` data format)
  or `(3, 299, 299)` (with `'channels_first'` data format).
  It should have exactly 3 inputs channels,
  and width and height should be no smaller than 75.
  E.g. `(150, 150, 3)` would be one valid value.
- **pooling**: Optional pooling mode for feature extraction
  when `include_top` is `False`.
  - `None` means that the output of the model will be
    the 4D tensor output of the last convolutional block.
  - `'avg'` means that global average pooling
    will be applied to the output of the
    last convolutional block, and thus
    the output of the model will be a 2D tensor.
  - `'max'` means that global max pooling will be applied.
- **classes**: optional number of classes to classify images
  into, only to be specified if `include_top` is `True`,
  and if no `weights` argument is specified.
- **classifier_activation**: A `str` or callable.
  The activation function to use on the "top" layer.
  Ignored unless `include_top=True`.
  Set `classifier_activation=None` to return the logits
  of the "top" layer. When loading pretrained weights,
  `classifier_activation` can only be `None` or `"softmax"`.
- **name**: The name of the model (string).

**Returns**

A model instance.
