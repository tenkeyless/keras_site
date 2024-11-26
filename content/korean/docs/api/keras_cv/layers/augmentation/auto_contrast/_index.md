---
title: AutoContrast layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/auto_contrast.py#L24" >}}

### `AutoContrast` class

```python
keras_cv.layers.AutoContrast(value_range, **kwargs)
```

Performs the AutoContrast operation on an image.

Auto contrast stretches the values of an image across the entire available
`value_range`. This makes differences between pixels more obvious. An
example of this is if an image only has values `[0, 1]` out of the range
`[0, 255]`, auto contrast will change the `1` values to be `255`.

**Arguments**

- **value_range**: the range of values the incoming images will have.
  Represented as a two number tuple written [low, high].
  This is typically either `[0, 1]` or `[0, 255]` depending
  on how your preprocessing pipeline is set up.
