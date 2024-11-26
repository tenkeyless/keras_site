---
title: RepeatVector layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/repeat_vector.py#L7" >}}

### `RepeatVector` class

```python
keras.layers.RepeatVector(n, **kwargs)
```

Repeats the input n times.

**Example**

```console
>>> x = keras.Input(shape=(32,))
>>> y = keras.layers.RepeatVector(3)(x)
>>> y.shape
(None, 3, 32)
```

**Arguments**

- **n**: Integer, repetition factor.

**Input shape**

2D tensor with shape `(batch_size, features)`.

**Output shape**

3D tensor with shape `(batch_size, n, features)`.
