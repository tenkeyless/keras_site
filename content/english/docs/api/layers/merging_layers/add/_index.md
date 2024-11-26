---
title: Add layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/merging/add.py#L6" >}}

### `Add` class

```python
keras.layers.Add(**kwargs)
```

Performs elementwise addition operation.

It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

**Examples**

```console
>>> input_shape = (2, 3, 4)
>>> x1 = np.random.rand(*input_shape)
>>> x2 = np.random.rand(*input_shape)
>>> y = keras.layers.Add()([x1, x2])
```

Usage in a Keras model:

```console
>>> input1 = keras.layers.Input(shape=(16,))
>>> x1 = keras.layers.Dense(8, activation='relu')(input1)
>>> input2 = keras.layers.Input(shape=(32,))
>>> x2 = keras.layers.Dense(8, activation='relu')(input2)
>>> # equivalent to `added = keras.layers.add([x1, x2])`
>>> added = keras.layers.Add()([x1, x2])
>>> out = keras.layers.Dense(4)(added)
>>> model = keras.models.Model(inputs=[input1, input2], outputs=out)
```