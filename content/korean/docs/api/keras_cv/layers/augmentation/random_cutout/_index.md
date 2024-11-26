---
title: RandomCutout layer
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_cutout.py#L28" >}}

### `RandomCutout` class

```python
keras_cv.layers.RandomCutout(
    height_factor, width_factor, fill_mode="constant", fill_value=0.0, seed=None, **kwargs
)
```

Randomly cut out rectangles from images and fill them.

**Arguments**

- **height_factor**: A tuple of two floats, a single float or a
  `keras_cv.FactorSampler`. `height_factor` controls the size of the
  cutouts. `height_factor=0.0` means the rectangle will be of size 0%
  of the image height, `height_factor=0.1` means the rectangle will
  have a size of 10% of the image height, and so forth. Values should
  be between `0.0` and `1.0`. If a tuple is used, a `height_factor`
  is sampled between the two values for every image augmented. If a
  single float is used, a value between `0.0` and the passed float is
  sampled. In order to ensure the value is always the same, please
  pass a tuple with two identical floats: `(0.5, 0.5)`.
- **width_factor**: A tuple of two floats, a single float or a
  `keras_cv.FactorSampler`. `width_factor` controls the size of the
  cutouts. `width_factor=0.0` means the rectangle will be of size 0%
  of the image height, `width_factor=0.1` means the rectangle will
  have a size of 10% of the image width, and so forth.
  Values should be between `0.0` and `1.0`. If a tuple is used, a
  `width_factor` is sampled between the two values for every image
  augmented. If a single float is used, a value between `0.0` and the
  passed float is sampled. In order to ensure the value is always the
  same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
- **fill_mode**: Pixels inside the patches are filled according to the given
  mode (one of `{"constant", "gaussian_noise"}`).
  - _constant_: Pixels are filled with the same constant value.
  - _gaussian_noise_: Pixels are filled with random gaussian noise.
- **fill_value**: a float represents the value to be filled inside the patches
  when `fill_mode="constant"`.
- **seed**: Integer. Used to create a random seed.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
random_cutout = keras_cv.layers.preprocessing.RandomCutout(0.5, 0.5)
augmented_images = random_cutout(images)
```
