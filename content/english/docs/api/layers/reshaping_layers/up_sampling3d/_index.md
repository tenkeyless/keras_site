---
title: UpSampling3D layer
toc: true
weight: 10
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/up_sampling3d.py#L9" >}}

### `UpSampling3D` class

```python
keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None, **kwargs)
```

Upsampling layer for 3D inputs.

Repeats the 1st, 2nd and 3rd dimensions of the data by `size[0]`, `size[1]` and `size[2]` respectively.

**Example**

```console
>>> input_shape = (2, 1, 2, 1, 3)
>>> x = np.ones(input_shape)
>>> y = keras.layers.UpSampling3D(size=(2, 2, 2))(x)
>>> y.shape
(2, 2, 4, 2, 3)
```

**Arguments**

- **size**: Int, or tuple of 3 integers. The upsampling factors for dim1, dim2 and dim3.
- **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `"channels_first"` corresponds to inputs with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists) else `"channels_last"`. Defaults to `"channels_last"`.

**Input shape**

5D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, dim1, dim2, dim3, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, dim1, dim2, dim3)`

**Output shape**

5D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
