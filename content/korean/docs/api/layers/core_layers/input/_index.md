---
title: Input object
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/core/input_layer.py#L95" >}}

### `Input` function

```python
keras.Input(
    shape=None,
    batch_size=None,
    dtype=None,
    sparse=None,
    batch_shape=None,
    name=None,
    tensor=None,
    optional=False,
)
```

Used to instantiate a Keras tensor.

A Keras tensor is a symbolic tensor-like object, which we augment with certain attributes that allow us to build a Keras model just by knowing the inputs and outputs of the model.

For instance, if `a`, `b` and `c` are Keras tensors, it becomes possible to do: `model = Model(input=[a, b], output=c)`

**Arguments**

- **shape**: A shape tuple (tuple of integers or `None` objects), not including the batch size. For instance, `shape=(32,)` indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be `None`; `None` elements represent dimensions where the shape is not known and may vary (e.g. sequence length).
- **batch_size**: Optional static batch size (integer).
- **dtype**: The data type expected by the input, as a string (e.g. `"float32"`, `"int32"`...)
- **sparse**: A boolean specifying whether the expected input will be sparse tensors. Note that, if `sparse` is `False`, sparse tensors can still be passed into the input - they will be densified with a default value of 0. This feature is only supported with the TensorFlow backend. Defaults to `False`.
- **batch_shape**: Optional shape tuple (tuple of integers or `None` objects), including the batch size.
- **name**: Optional name string for the layer. Should be unique in a model (do not reuse the same name twice). It will be autogenerated if it isn't provided.
- **tensor**: Optional existing tensor to wrap into the `Input` layer. If set, the layer will use this tensor rather than creating a new placeholder tensor.
- **optional**: Boolean, whether the input is optional or not. An optional input can accept `None` values.

**Returns**

A Keras tensor.

**Example**

```python
# This is a logistic regression in Keras
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```