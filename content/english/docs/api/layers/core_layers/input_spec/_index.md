---
title: InputSpec object
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/input_spec.py#L6" >}}

### `InputSpec` class

```python
keras.InputSpec(
    dtype=None,
    shape=None,
    ndim=None,
    max_ndim=None,
    min_ndim=None,
    axes=None,
    allow_last_axis_squeeze=False,
    name=None,
    optional=False,
)
```

Specifies the rank, dtype and shape of every input to a layer.

Layers can expose (if appropriate) an `input_spec` attribute: an instance of `InputSpec`, or a nested structure of `InputSpec` instances (one per input tensor). These objects enable the layer to run input compatibility checks for input structure, input rank, input shape, and input dtype for the first argument of `Layer.__call__`.

A `None` entry in a shape is compatible with any dimension.

**Arguments**

- **dtype**: Expected dtype of the input.
- **shape**: Shape tuple, expected shape of the input (may include `None` for dynamic axes). Includes the batch size.
- **ndim**: Integer, expected rank of the input.
- **max_ndim**: Integer, maximum rank of the input.
- **min_ndim**: Integer, minimum rank of the input.
- **axes**: Dictionary mapping integer axes to a specific dimension value.
- **allow_last_axis_squeeze**: If `True`, allow inputs of rank N+1 as long as the last axis of the input is 1, as well as inputs of rank N-1 as long as the last axis of the spec is 1.
- **name**: Expected key corresponding to this input when passing data as a dictionary.
- **optional**: Boolean, whether the input is optional or not. An optional input can accept `None` values.

**Example**

```python
class MyLayer(Layer):
    def __init__(self):
        super().__init__()
        # The layer will accept inputs with
        # shape (*, 28, 28) & (*, 28, 28, 1)
        # and raise an appropriate error message otherwise.
        self.input_spec = InputSpec(
            shape=(None, 28, 28, 1),
            allow_last_axis_squeeze=True)

```
