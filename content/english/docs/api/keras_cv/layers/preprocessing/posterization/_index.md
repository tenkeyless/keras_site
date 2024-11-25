---
title: Posterization layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/posterization.py#L24" >}}

### `Posterization` class

```python
keras_cv.layers.Posterization(value_range, bits, **kwargs)
```

Reduces the number of bits for each color channel.

**References**

- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)

**Arguments**

- **value_range**: a tuple or a list of two elements. The first value
  represents the lower bound for values in passed images, the second
  represents the upper bound. Images passed to the layer should have
  values within `value_range`. Defaults to `(0, 255)`.
- **bits**: integer, the number of bits to keep for each channel. Must be a
  value between 1-8.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
print(images[0, 0, 0])
# [59 62 63]
# Note that images are Tensors with values in the range [0, 255] and uint8
dtype
posterization = Posterization(bits=4, value_range=[0, 255])
images = posterization(images)
print(images[0, 0, 0])
# [48., 48., 48.]
# NOTE: the layer will output values in tf.float32, regardless of input
    dtype.
```

**Call arguments**

- **inputs**: input tensor in two possible formats:
  1. single 3D (HWC) image or 4D (NHWC) batch of images.
  2. A dict of tensors where the images are under `"images"` key.
