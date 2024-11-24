---
title: AutoContrast layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/image_preprocessing/auto_contrast.py#L9" >}}

### `AutoContrast` class

```python
keras.layers.AutoContrast(value_range=(0, 255), **kwargs)
```

Performs the auto-contrast operation on an image.

Auto contrast stretches the values of an image across the entire available `value_range`. This makes differences between pixels more obvious. An example of this is if an image only has values `[0, 1]` out of the range `[0, 255]`, auto contrast will change the `1` values to be `255`.

This layer is active at both training and inference time.

**Arguments**

- **value_range**: Range of values the incoming images will have. Represented as a two number tuple written `(low, high)`. This is typically either `(0, 1)` or `(0, 255)` depending on how your preprocessing pipeline is set up. Defaults to `(0, 255)`.
