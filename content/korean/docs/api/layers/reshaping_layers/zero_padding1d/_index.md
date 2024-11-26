---
title: ZeroPadding1D layer
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/zero_padding1d.py#L9" >}}

### `ZeroPadding1D` class

```python
keras.layers.ZeroPadding1D(padding=1, data_format=None, **kwargs)
```

Zero-padding layer for 1D input (e.g. temporal sequence).

**Example**

```console
>>> input_shape = (2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> x
[[[ 0  1  2]
  [ 3  4  5]]
 [[ 6  7  8]
  [ 9 10 11]]]
>>> y = keras.layers.ZeroPadding1D(padding=2)(x)
>>> y
[[[ 0  0  0]
  [ 0  0  0]
  [ 0  1  2]
  [ 3  4  5]
  [ 0  0  0]
  [ 0  0  0]]
 [[ 0  0  0]
  [ 0  0  0]
  [ 6  7  8]
  [ 9 10 11]
  [ 0  0  0]
  [ 0  0  0]]]
```

**Arguments**

- **padding**: Int, or tuple of int (length 2), or dictionary.
  - If int: how many zeros to add at the beginning and end of the padding dimension (axis 1).
  - If tuple of 2 ints: how many zeros to add at the beginning and the end of the padding dimension (`(left_pad, right_pad)`).
- **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch_size, axis_to_pad, channels)` while `"channels_first"` corresponds to inputs with shape `(batch_size, channels, axis_to_pad)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists). Defaults to `"channels_last"`.

**Input shape**

3D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, axis_to_pad, features)` - If `data_format` is `"channels_first"`: `(batch_size, features, axis_to_pad)`

**Output shape**

3D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, padded_axis, features)` - If `data_format` is `"channels_first"`: `(batch_size, features, padded_axis)`
