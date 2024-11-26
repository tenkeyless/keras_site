---
title: Solarization layer
toc: true
weight: 18
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/solarization.py#L25" >}}

### `Solarization` class

```python
keras_cv.layers.Solarization(
    value_range, addition_factor=0.0, threshold_factor=0.0, seed=None, **kwargs
)
```

Applies (max_value - pixel + min_value) for each pixel in the image.

When created without `threshold` parameter, the layer performs solarization
to all values. When created with specified `threshold` the layer only
augments pixels that are above the `threshold` value

**Reference**

- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
- [RandAugment](https://arxiv.org/pdf/1909.13719.pdf)

**Arguments**

- **value_range**: a tuple or a list of two elements. The first value
  represents the lower bound for values in passed images, the second
  represents the upper bound. Images passed to the layer should have
  values within `value_range`.
- **addition_factor**: (Optional) A tuple of two floats, a single float or a
  `keras_cv.FactorSampler`. For each augmented image a value is
  sampled from the provided range. If a float is passed, the range is
  interpreted as `(0, addition_factor)`. If specified, this value is
  added to each pixel before solarization and thresholding. The
  addition value should be scaled according to the value range
  (0, 255), defaults to 0.0.
- **threshold_factor**: (Optional) A tuple of two floats, a single float or
  a `keras_cv.FactorSampler`. For each augmented image a value is
  sampled from the provided range. If a float is passed, the range is
  interpreted as `(0, threshold_factor)`. If specified, only pixel
  values above this threshold will be solarized.
- **seed**: Integer. Used to create a random seed.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
print(images[0, 0, 0])
# [59 62 63]
# Note that images are Tensor with values in the range [0, 255]
solarization = Solarization(value_range=(0, 255))
images = solarization(images)
print(images[0, 0, 0])
# [196, 193, 192]
```

**Call arguments**

- **images**: Tensor of type int or float, with pixels in
  range [0, 255] and shape [batch, height, width, channels]
  or [height, width, channels].
