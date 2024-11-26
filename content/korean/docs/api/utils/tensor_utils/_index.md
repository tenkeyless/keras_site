---
title: Tensor utilities
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/operation_utils.py#L389" >}}

### `get_source_inputs` function

```python
keras.utils.get_source_inputs(tensor)
```

Returns the list of input tensors necessary to compute `tensor`.

Output will always be a list of tensors
(potentially with 1 element).

**Arguments**

- **tensor**: The tensor to start from.

**Returns**

List of input tensors.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/common/keras_tensor.py#L355" >}}

### `is_keras_tensor` function

```python
keras.utils.is_keras_tensor(x)
```

Returns whether `x` is a Keras tensor.

A "Keras tensor" is a _symbolic tensor_, such as a tensor
that was created via `Input()`. A "symbolic tensor"
can be understood as a placeholder – it does not
contain any actual numerical data, only a shape and dtype.
It can be used for building Functional models, but it
cannot be used in actual computations.
