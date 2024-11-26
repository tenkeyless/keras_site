---
title: Subtract layer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/merging/subtract.py#L6" >}}

### `Subtract` class

```python
keras.layers.Subtract(**kwargs)
```

Performs elementwise subtraction.

It takes as input a list of tensors of size 2 both of the same shape, and returns a single tensor (inputs[0] - inputs[1]) of same shape.

**Examples**

```console
>>> input_shape = (2, 3, 4)
>>> x1 = np.random.rand(*input_shape)
>>> x2 = np.random.rand(*input_shape)
>>> y = keras.layers.Subtract()([x1, x2])
```

Usage in a Keras model:

```console
>>> input1 = keras.layers.Input(shape=(16,))
>>> x1 = keras.layers.Dense(8, activation='relu')(input1)
>>> input2 = keras.layers.Input(shape=(32,))
>>> x2 = keras.layers.Dense(8, activation='relu')(input2)
>>> # equivalent to `subtracted = keras.layers.subtract([x1, x2])`
>>> subtracted = keras.layers.Subtract()([x1, x2])
>>> out = keras.layers.Dense(4)(subtracted)
>>> model = keras.models.Model(inputs=[input1, input2], outputs=out)
```
