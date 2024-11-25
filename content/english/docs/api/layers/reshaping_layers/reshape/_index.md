---
title: Reshape layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/reshape.py#L8" >}}

### `Reshape` class

```python
keras.layers.Reshape(target_shape, **kwargs)
```

Layer that reshapes inputs into the given shape.

**Arguments**

- **target_shape**: Target shape. Tuple of integers, does not include the samples dimension (batch size).

**Input shape**

Arbitrary, although all dimensions in the input shape must be known/fixed. Use the keyword argument `input_shape` (tuple of integers, does not include the samples/batch size axis) when using this layer as the first layer in a model.

**Output shape**

`(batch_size, *target_shape)`

**Example**

```console
>>> x = keras.Input(shape=(12,))
>>> y = keras.layers.Reshape((3, 4))(x)
>>> y.shape
(None, 3, 4)
```

```console
>>> # also supports shape inference using `-1` as dimension
>>> y = keras.layers.Reshape((-1, 2, 2))(x)
>>> y.shape
(None, 3, 2, 2)
```
