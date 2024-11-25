---
title: RandomSharpness layer
toc: true
weight: 16
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_sharpness.py#L24" >}}

### `RandomSharpness` class

```python
keras_cv.layers.RandomSharpness(factor, value_range, seed=None, **kwargs)
```

Randomly performs the sharpness operation on given images.

The sharpness operation first performs a blur operation, then blends between
the original image and the blurred image. This operation makes the edges of
an image less sharp than they were in the original image.

**References**

- [PIL](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)

**Arguments**

- **factor**: A tuple of two floats, a single float or
  `keras_cv.FactorSampler`. `factor` controls the extent to which the
  image sharpness is impacted. `factor=0.0` makes this layer perform a
  no-op operation, while a value of 1.0 uses the sharpened result
  entirely. Values between 0 and 1 result in linear interpolation
  between the original image and the sharpened image. Values should be
  between `0.0` and `1.0`. If a tuple is used, a `factor` is sampled
  between the two values for every image augmented. If a single float
  is used, a value between `0.0` and the passed float is sampled. In
  order to ensure the value is always the same, please pass a tuple
  with two identical floats: `(0.5, 0.5)`.
- **value_range**: the range of values the incoming images will have.
  Represented as a two number tuple written [low, high].
  This is typically either `[0, 1]` or `[0, 255]` depending
  on how your preprocessing pipeline is set up.
