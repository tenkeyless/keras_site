---
title: Activation layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/activations/activation.py#L6" >}}

### `Activation` class

```python
keras.layers.Activation(activation, **kwargs)
```

Applies an activation function to an output.

**Arguments**

- **activation**: Activation function. It could be a callable, or the name of an activation from the `keras.activations` namespace.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.

**Example**

```console
>>> layer = keras.layers.Activation('relu')
>>> layer([-3.0, -1.0, 0.0, 2.0])
[0.0, 0.0, 0.0, 2.0]
>>> layer = keras.layers.Activation(keras.activations.relu)
>>> layer([-3.0, -1.0, 0.0, 2.0])
[0.0, 0.0, 0.0, 2.0]
```
