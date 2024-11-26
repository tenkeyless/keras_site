---
title: Equalization layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/equalization.py#L26" >}}

### `Equalization` class

```python
keras_cv.layers.Equalization(value_range, bins=256, **kwargs)
```

Equalization performs histogram equalization on a channel-wise basis.

**Arguments**

- **value_range**: a tuple or a list of two elements. The first value
  represents the lower bound for values in passed images, the second
  represents the upper bound. Images passed to the layer should have
  values within `value_range`.
- **bins**: Integer indicating the number of bins to use in histogram
  equalization. Should be in the range [0, 256].

**Example**

```python
equalize = Equalization()
(images, labels), _ = keras.datasets.cifar10.load_data()
# Note that images are an int8 Tensor with values in the range [0, 255]
images = equalize(images)
```

**Call arguments**

- **images**: Tensor of pixels in range [0, 255], in RGB format. Can be
  of type float or int. Should be in NHWC format.
