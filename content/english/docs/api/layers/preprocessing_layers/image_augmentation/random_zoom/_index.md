---
title: RandomZoom layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/image_preprocessing/random_zoom.py#L9" >}}

### `RandomZoom` class

```python
keras.layers.RandomZoom(
    height_factor,
    width_factor=None,
    fill_mode="reflect",
    interpolation="bilinear",
    seed=None,
    fill_value=0.0,
    data_format=None,
    **kwargs
)
```

A preprocessing layer which randomly zooms images during training.

This layer will randomly zoom in or out on each axis of an image independently, filling empty space according to `fill_mode`.

Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype. By default, the layer will output floats.

**Input shape**

3D (unbatched) or 4D (batched) tensor with shape: `(..., height, width, channels)`, in `"channels_last"` format, or `(..., channels, height, width)`, in `"channels_first"` format.

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape: `(..., target_height, target_width, channels)`, or `(..., channels, target_height, target_width)`, in `"channels_first"` format.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Arguments**

- **height_factor**: a float represented as fraction of value, or a tuple of size 2 representing lower and upper bound for zooming vertically. When represented as a single float, this value is used for both the upper and lower bound. A positive value means zooming out, while a negative value means zooming in. For instance, `height_factor=(0.2, 0.3)` result in an output zoomed out by a random amount in the range `[+20%, +30%]`. `height_factor=(-0.3, -0.2)` result in an output zoomed in by a random amount in the range `[+20%, +30%]`.
- **width_factor**: a float represented as fraction of value, or a tuple of size 2 representing lower and upper bound for zooming horizontally. When represented as a single float, this value is used for both the upper and lower bound. For instance, `width_factor=(0.2, 0.3)` result in an output zooming out between 20% to 30%. `width_factor=(-0.3, -0.2)` result in an output zooming in between 20% to 30%. `None` means i.e., zooming vertical and horizontal directions by preserving the aspect ratio. Defaults to `None`.
- **fill_mode**: Points outside the boundaries of the input are filled according to the given mode. Available methods are `"constant"`, `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
  - `"reflect"`: `(d c b a | a b c d | d c b a)` The input is extended by reflecting about the edge of the last pixel.
  - `"constant"`: `(k k k k | a b c d | k k k k)` The input is extended by filling all values beyond the edge with the same constant value k specified by `fill_value`.
  - `"wrap"`: `(a b c d | a b c d | a b c d)` The input is extended by wrapping around to the opposite edge.
  - `"nearest"`: `(a a a a | a b c d | d d d d)` The input is extended by the nearest pixel. Note that when using torch backend, `"reflect"` is redirected to `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not support `"reflect"`. Note that torch backend does not support `"wrap"`.
- **interpolation**: Interpolation mode. Supported values: `"nearest"`, `"bilinear"`.
- **seed**: Integer. Used to create a random seed.
- **fill_value**: a float that represents the value to be filled outside the boundaries when `fill_mode="constant"`.
- **data_format**: string, either `"channels_last"` or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, height, width, channels)` while `"channels_first"` corresponds to inputs with shape `(batch, channels, height, width)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. If you never set it, then it will be `"channels_last"`.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.

**Example**

```console
>>> input_img = np.random.random((32, 224, 224, 3))
>>> layer = keras.layers.RandomZoom(.5, .2)
>>> out_img = layer(input_img)
```
