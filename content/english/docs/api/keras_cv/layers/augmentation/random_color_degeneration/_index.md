---
title: RandomColorDegeneration layer
toc: true
weight: 12
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_color_degeneration.py#L25" >}}

### `RandomColorDegeneration` class

```python
keras_cv.layers.RandomColorDegeneration(factor, seed=None, **kwargs)
```

Randomly performs the color degeneration operation on given images.

The sharpness operation first converts an image to gray scale, then back to
color. It then takes a weighted average between original image and the
degenerated image. This makes colors appear more dull.

**Arguments**

- **factor**: A tuple of two floats, a single float or a
  `keras_cv.FactorSampler`. `factor` controls the extent to which the
  image sharpness is impacted. `factor=0.0` makes this layer perform a
  no-op operation, while a value of 1.0 uses the degenerated result
  entirely. Values between 0 and 1 result in linear interpolation
  between the original image and the sharpened image.
  Values should be between `0.0` and `1.0`. If a tuple is used, a
  `factor` is sampled between the two values for every image
  augmented. If a single float is used, a value between `0.0` and the
  passed float is sampled. In order to ensure the value is always the
  same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
- **seed**: Integer. Used to create a random seed.
