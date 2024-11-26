---
title: RandomHue layer
toc: true
weight: 14
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_hue.py#L25" >}}

### `RandomHue` class

```python
keras_cv.layers.RandomHue(factor, value_range, seed=None, **kwargs)
```

Randomly adjusts the hue on given images.

This layer will randomly increase/reduce the hue for the input RGB
images.

The image hue is adjusted by converting the image(s) to HSV and rotating the
hue channel (H) by delta. The image is then converted back to RGB.

**Arguments**

- **factor**: A tuple of two floats, a single float or
  `keras_cv.FactorSampler`. `factor` controls the extent to which the
  image hue is impacted. `factor=0.0` makes this layer perform a
  no-op operation, while a value of 1.0 performs the most aggressive
  contrast adjustment available. If a tuple is used, a `factor` is
  sampled between the two values for every image augmented. If a
  single float is used, a value between `0.0` and the passed float is
  sampled. In order to ensure the value is always the same, please
  pass a tuple with two identical floats: `(0.5, 0.5)`.
- **value_range**: the range of values the incoming images will have.
  Represented as a two number tuple written [low, high]. This is
  typically either `[0, 1]` or `[0, 255]` depending on how your
  preprocessing pipeline is set up.
- **seed**: Integer. Used to create a random seed.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
random_hue = keras_cv.layers.preprocessing.RandomHue()
augmented_images = random_hue(images)
```
