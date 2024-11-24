---
title: RandomRotation layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/image_preprocessing/random_rotation.py#L10" >}}

### `RandomRotation` class

```python
keras.layers.RandomRotation(
    factor,
    fill_mode="reflect",
    interpolation="bilinear",
    seed=None,
    fill_value=0.0,
    data_format=None,
    **kwargs
)
```

A preprocessing layer which randomly rotates images during training.

This layer will apply random rotations to each image, filling empty space according to `fill_mode`.

By default, random rotations are only applied during training. At inference time, the layer does nothing. If you need to apply random rotations at inference time, pass `training=True` when calling the layer.

Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype. By default, the layer will output floats.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Input shape**

3D (unbatched) or 4D (batched) tensor with shape: `(..., height, width, channels)`, in `"channels_last"` format

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape: `(..., height, width, channels)`, in `"channels_last"` format

**Arguments**

- **factor**: a float represented as fraction of 2 Pi, or a tuple of size 2 representing lower and upper bound for rotating clockwise and counter-clockwise. A positive values means rotating counter clock-wise, while a negative value means clock-wise. When represented as a single float, this value is used for both the upper and lower bound. For instance, `factor=(-0.2, 0.3)` results in an output rotation by a random amount in the range `[-20% * 2pi, 30% * 2pi]`. `factor=0.2` results in an output rotating by a random amount in the range `[-20% * 2pi, 20% * 2pi]`.
- **fill_mode**: Points outside the boundaries of the input are filled according to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
  - _reflect_: `(d c b a | a b c d | d c b a)` The input is extended by reflecting about the edge of the last pixel.
  - _constant_: `(k k k k | a b c d | k k k k)` The input is extended by filling all values beyond the edge with the same constant value k = 0.
  - _wrap_: `(a b c d | a b c d | a b c d)` The input is extended by wrapping around to the opposite edge.
  - _nearest_: `(a a a a | a b c d | d d d d)` The input is extended by the nearest pixel.
- **interpolation**: Interpolation mode. Supported values: `"nearest"`, `"bilinear"`.
- **seed**: Integer. Used to create a random seed.
- **fill_value**: a float represents the value to be filled outside the boundaries when `fill_mode="constant"`.
- **data_format**: string, either `"channels_last"` or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, height, width, channels)` while `"channels_first"` corresponds to inputs with shape `(batch, channels, height, width)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. If you never set it, then it will be `"channels_last"`.
