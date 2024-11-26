---
title: RandomShear layer
toc: true
weight: 17
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_shear.py#L27" >}}

### `RandomShear` class

```python
keras_cv.layers.RandomShear(
    x_factor=None,
    y_factor=None,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    bounding_box_format=None,
    seed=None,
    **kwargs
)
```

A preprocessing layer which randomly shears images.

This layer will apply random shearings to each image, filling empty space
according to `fill_mode`.

Input pixel values can be of any range and any data type.

**Input shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format

**Arguments**

- **x_factor**: A tuple of two floats, a single float or a
  `keras_cv.FactorSampler`. For each augmented image a value is
  sampled from the provided range. If a float is passed, the range is
  interpreted as `(0, x_factor)`. Values represent a percentage of the
  image to shear over. For example, 0.3 shears pixels up to 30% of the
  way across the image. All provided values should be positive. If
  `None` is passed, no shear occurs on the X axis. Defaults to `None`.
- **y_factor**: A tuple of two floats, a single float or a
  `keras_cv.FactorSampler`. For each augmented image a value is
  sampled from the provided range. If a float is passed, the range is
  interpreted as `(0, y_factor)`. Values represent a percentage of the
  image to shear over. For example, 0.3 shears pixels up to 30% of the
  way across the image. All provided values should be positive. If
  `None` is passed, no shear occurs on the Y axis. Defaults to `None`.
- **interpolation**: interpolation method used in the
  `ImageProjectiveTransformV3` op. Supported values are `"nearest"`
  and `"bilinear"`, defaults to `"bilinear"`.
- **fill_mode**: fill_mode in the `ImageProjectiveTransformV3` op. Supported
  values are `"reflect"`, `"wrap"`, `"constant"`, and `"nearest"`.
  Defaults to `"reflect"`.
- **fill_value**: fill_value in the `ImageProjectiveTransformV3` op. A
  `Tensor` of type `float32`. The value to be filled when fill_mode is
  constant". Defaults to `0.0`.
- **bounding_box_format**: The format of bounding boxes of input dataset.
  Refer to
  https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
  for more details on supported bounding box formats.
- **seed**: Integer. Used to create a random seed.
