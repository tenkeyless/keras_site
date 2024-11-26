---
title: ZeroPadding3D layer
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/zero_padding3d.py#L9" >}}

### `ZeroPadding3D` class

```python
keras.layers.ZeroPadding3D(
    padding=((1, 1), (1, 1), (1, 1)), data_format=None, **kwargs
)
```

Zero-padding layer for 3D data (spatial or spatio-temporal).

**Example**

```console
>>> input_shape = (1, 1, 2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> y = keras.layers.ZeroPadding3D(padding=2)(x)
>>> y.shape
(1, 5, 6, 6, 3)
```

**Arguments**

- **padding**: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
  - If int: the same symmetric padding is applied to depth, height, and width.
  - If tuple of 3 ints: interpreted as three different symmetric padding values for depth, height, and width: `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
  - If tuple of 3 tuples of 2 ints: interpreted as `((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))`.
- **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `"channels_first"` corresponds to inputs with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists). Defaults to `"channels_last"`.

**Input shape**

5D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth)` - If `data_format` is `"channels_first"`: `(batch_size, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`

**Output shape**

5D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)` - If `data_format` is `"channels_first"`: `(batch_size, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`
