---
title: SqueezeAndExcite2D layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/regularization/squeeze_excite.py#L20" >}}

### `SqueezeAndExcite2D` class

```python
keras_cv.layers.SqueezeAndExcite2D(
    filters,
    bottleneck_filters=None,
    squeeze_activation="relu",
    excite_activation="sigmoid",
    **kwargs
)
```

Implements Squeeze and Excite block as in
[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf).
This layer tries to use a content aware mechanism to assign channel-wise
weights adaptively. It first squeezes the feature maps into a single value
using global average pooling, which are then fed into two Conv1D layers,
which act like fully-connected layers. The first layer reduces the
dimensionality of the feature maps, and second layer restores it to its
original value.

The resultant values are the adaptive weights for each channel. These
weights are then multiplied with the original inputs to scale the outputs
based on their individual weightages.

**Arguments**

- **filters**: Number of input and output filters. The number of input and
  output filters is same.
- **bottleneck_filters**: (Optional) Number of bottleneck filters. Defaults
  to `0.25 * filters`
- **squeeze_activation**: (Optional) String, callable (or
  keras.layers.Layer) or keras.activations.Activation instance
  denoting activation to be applied after squeeze convolution.
  Defaults to `relu`.
- **excite_activation**: (Optional) String, callable (or
  keras.layers.Layer) or keras.activations.Activation instance
  denoting activation to be applied after excite convolution.
  Defaults to `sigmoid`.

**Example**

```python
# (...)
input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
x = keras.layers.Conv2D(16, (3, 3))(input)
output = keras_cv.layers.SqueezeAndExciteBlock(16)(x)
# (...)
```
