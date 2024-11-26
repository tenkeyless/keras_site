---
title: RandomSaturation layer
toc: true
weight: 15
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/random_saturation.py#L25" >}}

### `RandomSaturation` class

```python
keras_cv.layers.RandomSaturation(factor, seed=None, **kwargs)
```

Randomly adjusts the saturation on given images.

This layer will randomly increase/reduce the saturation for the input RGB
images.

**Arguments**

- **factor**: A tuple of two floats, a single float or
  `keras_cv.FactorSampler`. `factor` controls the extent to which the
  image saturation is impacted. `factor=0.5` makes this layer perform
  a no-op operation. `factor=0.0` makes the image to be fully
  grayscale. `factor=1.0` makes the image to be fully saturated.
  Values should be between `0.0` and `1.0`. If a tuple is used, a
  `factor` is sampled between the two values for every image
  augmented. If a single float is used, a value between `0.0` and the
  passed float is sampled. In order to ensure the value is always the
  same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
- **seed**: Integer. Used to create a random seed.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
random_saturation = keras_cv.layers.preprocessing.RandomSaturation()
augmented_images = random_saturation(images)
```
