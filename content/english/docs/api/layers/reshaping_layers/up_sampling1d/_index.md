---
title: UpSampling1D layer
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/up_sampling1d.py#L7" >}}

### `UpSampling1D` class

```python
keras.layers.UpSampling1D(size=2, **kwargs)
```

Upsampling layer for 1D inputs.

Repeats each temporal step `size` times along the time axis.

**Example**

```console
>>> input_shape = (2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> x
[[[ 0  1  2]
  [ 3  4  5]]
 [[ 6  7  8]
  [ 9 10 11]]]
>>> y = keras.layers.UpSampling1D(size=2)(x)
>>> y
[[[ 0.  1.  2.]
  [ 0.  1.  2.]
  [ 3.  4.  5.]
  [ 3.  4.  5.]]
 [[ 6.  7.  8.]
  [ 6.  7.  8.]
  [ 9. 10. 11.]
  [ 9. 10. 11.]]]
```

**Arguments**

- **size**: Integer. Upsampling factor.

**Input shape**

3D tensor with shape: `(batch_size, steps, features)`.

**Output shape**

3D tensor with shape: `(batch_size, upsampled_steps, features)`.
