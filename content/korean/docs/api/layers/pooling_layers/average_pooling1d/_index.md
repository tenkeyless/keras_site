---
title: AveragePooling1D layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/pooling/average_pooling1d.py#L5" >}}

### `AveragePooling1D` class

```python
keras.layers.AveragePooling1D(
    pool_size, strides=None, padding="valid", data_format=None, name=None, **kwargs
)
```

Average pooling for temporal data.

Downsamples the input representation by taking the average value over the window defined by `pool_size`. The window is shifted by `strides`. The resulting output when using "valid" padding option has a shape of: `output_shape = (input_shape - pool_size + 1) / strides)`

The resulting output shape when using the "same" padding option is: `output_shape = input_shape / strides`

**Arguments**

- **pool_size**: int, size of the max pooling window.
- **strides**: int or None. Specifies how much the pooling window moves for each pooling step. If None, it will default to `pool_size`.
- **padding**: string, either `"valid"` or `"same"` (case-insensitive). `"valid"` means no padding. `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
- **data_format**: string, either `"channels_last"` or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, steps, features)` while `"channels_first"` corresponds to inputs with shape `(batch, features, steps)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. If you never set it, then it will be `"channels_last"`.

**Input shape**

- If `data_format="channels_last"`: 3D tensor with shape `(batch_size, steps, features)`.
- If `data_format="channels_first"`: 3D tensor with shape `(batch_size, features, steps)`.

**Output shape**

- If `data_format="channels_last"`: 3D tensor with shape `(batch_size, downsampled_steps, features)`.
- If `data_format="channels_first"`: 3D tensor with shape `(batch_size, features, downsampled_steps)`.

**Examples**

`strides=1` and `padding="valid"`:

```console
>>> x = np.array([1., 2., 3., 4., 5.])
>>> x = np.reshape(x, [1, 5, 1])
>>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,
...    strides=1, padding="valid")
>>> avg_pool_1d(x)
```

`strides=2` and `padding="valid"`:

```console
>>> x = np.array([1., 2., 3., 4., 5.])
>>> x = np.reshape(x, [1, 5, 1])
>>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,
...    strides=2, padding="valid")
>>> avg_pool_1d(x)
```

`strides=1` and `padding="same"`:

```console
>>> x = np.array([1., 2., 3., 4., 5.])
>>> x = np.reshape(x, [1, 5, 1])
>>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,
...    strides=1, padding="same")
>>> avg_pool_1d(x)
```
