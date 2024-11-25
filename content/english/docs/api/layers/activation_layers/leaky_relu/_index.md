---
title: LeakyReLU layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/activations/leaky_relu.py#L8" >}}

### `LeakyReLU` class

```python
keras.layers.LeakyReLU(negative_slope=0.3, **kwargs)
```

Leaky version of a Rectified Linear Unit activation layer.

This layer allows a small gradient when the unit is not active.

Formula:

```python
f(x) = alpha * x if x < 0
f(x) = x if x >= 0
```

**Example**

```python
leaky_relu_layer = LeakyReLU(negative_slope=0.5)
input = np.array([-10, -5, 0.0, 5, 10])
result = leaky_relu_layer(input)
```

**Arguments**

- **negative_slope**: Float >= 0.0. Negative slope coefficient.
  Defaults to `0.3`.
- **\*\*kwargs**: Base layer keyword arguments, such as
  `name` and `dtype`.
