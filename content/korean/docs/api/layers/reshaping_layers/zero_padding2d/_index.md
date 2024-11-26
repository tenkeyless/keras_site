---
title: ZeroPadding2D layer
toc: true
weight: 12
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/zero_padding2d.py#L9" >}}

### `ZeroPadding2D` class

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None, **kwargs)
```

Zero-padding layer for 2D input (e.g. picture).

This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.

**Example**

```console
>>> input_shape = (1, 1, 2, 2)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> x
[[[[0 1]
   [2 3]]]]
>>> y = keras.layers.ZeroPadding2D(padding=1)(x)
>>> y
[[[[0 0]
   [0 0]
   [0 0]
   [0 0]]
  [[0 0]
   [0 1]
   [2 3]
   [0 0]]
  [[0 0]
   [0 0]
   [0 0]
   [0 0]]]]
```

**Arguments**

- **padding**: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
  - If int: the same symmetric padding is applied to height and width.
  - If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_height_pad, symmetric_width_pad)`.
  - If tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad, right_pad))`.
- **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch_size, height, width, channels)` while `"channels_first"` corresponds to inputs with shape `(batch_size, channels, height, width)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists). Defaults to `"channels_last"`.

**Input shape**

4D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, height, width, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, height, width)`

**Output shape**

4D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, padded_height, padded_width, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, padded_height, padded_width)`
