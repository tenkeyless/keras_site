---
title: Average layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/merging/average.py#L6" >}}

### `Average` class

```python
keras.layers.Average(**kwargs)
```

Averages a list of inputs element-wise..

It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

**Examples**

```console
>>> input_shape = (2, 3, 4)
>>> x1 = np.random.rand(*input_shape)
>>> x2 = np.random.rand(*input_shape)
>>> y = keras.layers.Average()([x1, x2])
```

Usage in a Keras model:

```console
>>> input1 = keras.layers.Input(shape=(16,))
>>> x1 = keras.layers.Dense(8, activation='relu')(input1)
>>> input2 = keras.layers.Input(shape=(32,))
>>> x2 = keras.layers.Dense(8, activation='relu')(input2)
>>> # equivalent to `y = keras.layers.average([x1, x2])`
>>> y = keras.layers.Average()([x1, x2])
>>> out = keras.layers.Dense(4)(y)
>>> model = keras.models.Model(inputs=[input1, input2], outputs=out)
```
