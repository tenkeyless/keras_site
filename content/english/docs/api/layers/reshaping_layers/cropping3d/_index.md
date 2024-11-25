---
title: Cropping3D layer
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/cropping3d.py#L8" >}}

### `Cropping3D` class

```python
keras.layers.Cropping3D(
    cropping=((1, 1), (1, 1), (1, 1)), data_format=None, **kwargs
)
```

Cropping layer for 3D data (e.g. spatial or spatio-temporal).

**Example**

```console
>>> input_shape = (2, 28, 28, 10, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> y = keras.layers.Cropping3D(cropping=(2, 4, 2))(x)
>>> y.shape
(2, 24, 20, 6, 3)
```

**Arguments**

- **cropping**: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
  - If int: the same symmetric cropping is applied to depth, height, and width.
  - If tuple of 3 ints: interpreted as three different symmetric cropping values for depth, height, and width: `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
  - If tuple of 3 tuples of 2 ints: interpreted as `((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`.
- **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `"channels_first"` corresponds to inputs with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists). Defaults to `"channels_last"`.

**Input shape**

5D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`

**Output shape**

5D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, first_cropped_axis, second_cropped_axis, third_cropped_axis, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, first_cropped_axis, second_cropped_axis, third_cropped_axis)`
