---
title: HyperXception
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/applications/xception.py#L23" >}}

### `HyperXception` class

```python
keras_tuner.applications.HyperXception(
    include_top=True, input_shape=None, input_tensor=None, classes=None, **kwargs
)
```

An Xception hypermodel.

Models built by `HyperXception` take images with shape (height, width,
channels) as input. The output are one-hot encoded with the length matching
the number of classes specified by the `classes` argument.

**Arguments**

- **include_top**: Boolean, whether to include the fully-connected layer at
  the top of the network.
- **input_shape**: Optional shape tuple, e.g. `(256, 256, 3)`. One of
  `input_shape` or `input_tensor` must be specified.
- **input_tensor**: Optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model. One of `input_shape` or
  `input_tensor` must be specified.
- **classes**: Optional number of classes to classify images into, only to be
  specified if `include_top` is True, and if no `weights` argument is
  specified.
- **\*\*kwargs**: Additional keyword arguments that apply to all hypermodels.
  See [`keras_tuner.HyperModel`]({{< relref "/docs/api/keras_tuner/hypermodels/base_hypermodel#hypermodel-class" >}}).
