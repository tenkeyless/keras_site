---
title: SpatialDropout1D layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/regularization/spatial_dropout.py#L31" >}}

### `SpatialDropout1D` class

```python
keras.layers.SpatialDropout1D(rate, seed=None, name=None, dtype=None)
```

Spatial 1D version of Dropout.

This layer performs the same function as Dropout, however, it drops entire 1D feature maps instead of individual elements. If adjacent frames within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, `SpatialDropout1D` will help promote independence between feature maps and should be used instead.

**Arguments**

- **rate**: Float between 0 and 1. Fraction of the input units to drop.

**Call arguments**

- **inputs**: A 3D tensor.
- **training**: Python boolean indicating whether the layer should behave in training mode (applying dropout) or in inference mode (pass-through).

**Input shape**

3D tensor with shape: `(samples, timesteps, channels)`

**Output shape Same as input.**

**Reference**

- [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)
