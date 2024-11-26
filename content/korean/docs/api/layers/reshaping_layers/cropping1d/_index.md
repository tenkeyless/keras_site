---
title: Cropping1D layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/cropping1d.py#L7" >}}

### `Cropping1D` class

```python
keras.layers.Cropping1D(cropping=(1, 1), **kwargs)
```

Cropping layer for 1D input (e.g. temporal sequence).

It crops along the time dimension (axis 1).

**Example**

```console
>>> input_shape = (2, 3, 2)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> x
[[[ 0  1]
  [ 2  3]
  [ 4  5]]
 [[ 6  7]
  [ 8  9]
  [10 11]]]
>>> y = keras.layers.Cropping1D(cropping=1)(x)
>>> y
[[[2 3]]
 [[8 9]]]
```

**Arguments**

- **cropping**: Int, or tuple of int (length 2), or dictionary.
  - If int: how many units should be trimmed off at the beginning and end of the cropping dimension (axis 1).
  - If tuple of 2 ints: how many units should be trimmed off at the beginning and end of the cropping dimension (`(left_crop, right_crop)`).

**Input shape**

3D tensor with shape `(batch_size, axis_to_crop, features)`

**Output shape**

3D tensor with shape `(batch_size, cropped_axis, features)`
