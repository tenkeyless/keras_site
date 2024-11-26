---
title: ActivityRegularization layer
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/regularization/activity_regularization.py#L6" >}}

### `ActivityRegularization` class

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0, **kwargs)
```

Layer that applies an update to the cost function based input activity.

**Arguments**

- **l1**: L1 regularization factor (positive float).
- **l2**: L2 regularization factor (positive float).

**Input shape**

Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

**Output shape**

Same shape as input.
