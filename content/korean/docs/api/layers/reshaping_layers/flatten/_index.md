---
title: Flatten layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/flatten.py#L11" >}}

### `Flatten` class

```python
keras.layers.Flatten(data_format=None, **kwargs)
```

Flattens the input. Does not affect the batch size.

Note: If inputs are shaped `(batch,)` without a feature axis, then flattening adds an extra channel dimension and output shape is `(batch, 1)`.

**Arguments**

- **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, ..., channels)` while `"channels_first"` corresponds to inputs with shape `(batch, channels, ...)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists). Defaults to `"channels_last"`.

**Example**

```console
>>> x = keras.Input(shape=(10, 64))
>>> y = keras.layers.Flatten()(x)
>>> y.shape
(None, 640)
```
