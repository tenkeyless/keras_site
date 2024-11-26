---
title: PReLU layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/activations/prelu.py#L10" >}}

### `PReLU` class

```python
keras.layers.PReLU(
    alpha_initializer="Zeros",
    alpha_regularizer=None,
    alpha_constraint=None,
    shared_axes=None,
    **kwargs
)
```

Parametric Rectified Linear Unit activation layer.

Formula:

```python
f(x) = alpha * x for x < 0
f(x) = x for x >= 0
```

where `alpha` is a learned array with the same shape as x.

**Arguments**

- **alpha_initializer**: Initializer function for the weights.
- **alpha_regularizer**: Regularizer for the weights.
- **alpha_constraint**: Constraint for the weights.
- **shared_axes**: The axes along which to share learnable parameters for the
  activation function. For example, if the incoming feature maps are
  from a 2D convolution with output shape
  `(batch, height, width, channels)`, and you wish to share parameters
  across space so that each filter only has one set of parameters,
  set `shared_axes=[1, 2]`.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.
