---
title: Cropping2D layer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/cropping2d.py#L8" >}}

### `Cropping2D` class

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None, **kwargs)
```

Cropping layer for 2D input (e.g. picture).

It crops along spatial dimensions, i.e. height and width.

**Example**

```console
>>> input_shape = (2, 28, 28, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> y = keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
>>> y.shape
(2, 24, 20, 3)
```

**Arguments**

- **cropping**: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
  - If int: the same symmetric cropping is applied to height and width.
  - If tuple of 2 ints: interpreted as two different symmetric cropping values for height and width: `(symmetric_height_crop, symmetric_width_crop)`.
  - If tuple of 2 tuples of 2 ints: interpreted as `((top_crop, bottom_crop), (left_crop, right_crop))`.
- **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch_size, height, width, channels)` while `"channels_first"` corresponds to inputs with shape `(batch_size, channels, height, width)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists). Defaults to `"channels_last"`.

**Input shape**

4D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, height, width, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, height, width)`

**Output shape**

4D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, cropped_height, cropped_width, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, cropped_height, cropped_width)`
