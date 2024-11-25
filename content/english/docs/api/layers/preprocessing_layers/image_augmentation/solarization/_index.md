---
title: Solarization layer
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/image_preprocessing/solarization.py#L10" >}}

### `Solarization` class

```python
keras.layers.Solarization(
    addition_factor=0.0, threshold_factor=0.0, value_range=(0, 255), seed=None, **kwargs
)
```

Applies `(max_value - pixel + min_value)` for each pixel in the image.

When created without `threshold` parameter, the layer performs solarization to all values. When created with specified `threshold` the layer only augments pixels that are above the `threshold` value.

**Arguments**

- **addition_factor**: (Optional) A tuple of two floats or a single float, between 0 and 1. For each augmented image a value is sampled from the provided range. If a float is passed, the range is interpreted as `(0, addition_factor)`. If specified, this value (times the value range of input images, e.g. 255), is added to each pixel before solarization and thresholding. Defaults to 0.0.
- **threshold_factor**: (Optional) A tuple of two floats or a single float. For each augmented image a value is sampled from the provided range. If a float is passed, the range is interpreted as `(0, threshold_factor)`. If specified, only pixel values above this threshold will be solarized.
- **value_range**: a tuple or a list of two elements. The first value represents the lower bound for values in input images, the second represents the upper bound. Images passed to the layer should have values within `value_range`. Typical values to pass are `(0, 255)` (RGB image) or `(0., 1.)` (scaled image).
- **seed**: Integer. Used to create a random seed.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.

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
