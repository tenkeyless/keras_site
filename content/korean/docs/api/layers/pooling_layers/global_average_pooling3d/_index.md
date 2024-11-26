---
title: GlobalAveragePooling3D layer
toc: true
weight: 12
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/pooling/global_average_pooling3d.py#L6" >}}

### `GlobalAveragePooling3D` class

```python
keras.layers.GlobalAveragePooling3D(data_format=None, keepdims=False, **kwargs)
```

Global average pooling operation for 3D data.

**Arguments**

- **data_format**: string, either `"channels_last"` or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `"channels_first"` corresponds to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. If you never set it, then it will be `"channels_last"`.
- **keepdims**: A boolean, whether to keep the temporal dimension or not. If `keepdims` is `False` (default), the rank of the tensor is reduced for spatial dimensions. If `keepdims` is `True`, the spatial dimension are retained with length 1. The behavior is the same as for [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/reduce_mean) or `np.mean`.

**Input shape**

- If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

**Output shape**

- If `keepdims=False`: 2D tensor with shape `(batch_size, channels)`.
- If `keepdims=True`: - If `data_format="channels_last"`: 5D tensor with shape `(batch_size, 1, 1, 1, channels)` - If `data_format="channels_first"`: 5D tensor with shape `(batch_size, channels, 1, 1, 1)`

**Example**

```console
>>> x = np.random.rand(2, 4, 5, 4, 3)
>>> y = keras.layers.GlobalAveragePooling3D()(x)
>>> y.shape
(2, 3)
```
