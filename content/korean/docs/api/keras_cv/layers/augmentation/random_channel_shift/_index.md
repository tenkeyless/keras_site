---
title: RandomChannelShift layer
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_channel_shift.py#L24" >}}

### `RandomChannelShift` class

```python
keras_cv.layers.RandomChannelShift(
    value_range, factor, channels=3, seed=None, **kwargs
)
```

Randomly shift values for each channel of the input image(s).

The input images should have values in the `[0-255]` or `[0-1]` range.

**Input shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `channels_last` format.

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `channels_last` format.

**Arguments**

- **value_range**: The range of values the incoming images will have.
  Represented as a two number tuple written [low, high].
  This is typically either `[0, 1]` or `[0, 255]` depending
  on how your preprocessing pipeline is set up.
- **factor**: A scalar value, or tuple/list of two floating values in
  the range `[0.0, 1.0]`. If `factor` is a single value, it will
  interpret as equivalent to the tuple `(0.0, factor)`. The `factor`
  will sample between its range for every image to augment.
- **channels**: integer, the number of channels to shift, defaults to 3 which
  corresponds to an RGB shift. In some cases, there may ber more or
  less channels.
- **seed**: Integer. Used to create a random seed.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
rgb_shift = keras_cv.layers.RandomChannelShift(value_range=(0, 255),
    factor=0.5)
augmented_images = rgb_shift(images)
```
