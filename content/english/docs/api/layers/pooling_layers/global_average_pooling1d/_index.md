---
title: GlobalAveragePooling1D layer
toc: true
weight: 10
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/pooling/global_average_pooling1d.py#L7" >}}

### `GlobalAveragePooling1D` class

```python
keras.layers.GlobalAveragePooling1D(data_format=None, keepdims=False, **kwargs)
```

Global average pooling operation for temporal data.

**Arguments**

- **data_format**: string, either `"channels_last"` or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, steps, features)` while `"channels_first"` corresponds to inputs with shape `(batch, features, steps)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. If you never set it, then it will be `"channels_last"`.
- **keepdims**: A boolean, whether to keep the temporal dimension or not. If `keepdims` is `False` (default), the rank of the tensor is reduced for spatial dimensions. If `keepdims` is `True`, the temporal dimension are retained with length 1. The behavior is the same as for [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/reduce_mean) or `np.mean`.

**Call arguments**

- **inputs**: A 3D tensor.
- **mask**: Binary tensor of shape `(batch_size, steps)` indicating whether a given step should be masked (excluded from the average).

**Input shape**

- If `data_format='channels_last'`: 3D tensor with shape: `(batch_size, steps, features)`
- If `data_format='channels_first'`: 3D tensor with shape: `(batch_size, features, steps)`

**Output shape**

- If `keepdims=False`: 2D tensor with shape `(batch_size, features)`.
- If `keepdims=True`: - If `data_format="channels_last"`: 3D tensor with shape `(batch_size, 1, features)` - If `data_format="channels_first"`: 3D tensor with shape `(batch_size, features, 1)`

**Example**

```console
>>> x = np.random.rand(2, 3, 4)
>>> y = keras.layers.GlobalAveragePooling1D()(x)
>>> y.shape
(2, 4)
```
