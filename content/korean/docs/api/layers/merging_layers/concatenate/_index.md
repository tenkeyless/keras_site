---
title: Concatenate layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/merging/concatenate.py#L6" >}}

### `Concatenate` class

```python
keras.layers.Concatenate(axis=-1, **kwargs)
```

Concatenates a list of inputs.

It takes as input a list of tensors, all of the same shape except for the concatenation axis, and returns a single tensor that is the concatenation of all inputs.

**Examples**

```console
>>> x = np.arange(20).reshape(2, 2, 5)
>>> y = np.arange(20, 30).reshape(2, 1, 5)
>>> keras.layers.Concatenate(axis=1)([x, y])
```

Usage in a Keras model:

```console
>>> x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
>>> x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
>>> y = keras.layers.Concatenate()([x1, x2])
```

**Arguments**

- **axis**: Axis along which to concatenate.
- **\*\*kwargs**: Standard layer keyword arguments.

**Returns**

A tensor, the concatenation of the inputs alongside axis `axis`.
