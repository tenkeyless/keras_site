---
title: GaussianNoise layer
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/regularization/gaussian_noise.py#L7" >}}

### `GaussianNoise` class

```python
keras.layers.GaussianNoise(stddev, seed=None, **kwargs)
```

Apply additive zero-centered Gaussian noise.

This is useful to mitigate overfitting (you could see it as a form of random data augmentation). Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.

As it is a regularization layer, it is only active at training time.

**Arguments**

- **stddev**: Float, standard deviation of the noise distribution.
- **seed**: Integer, optional random seed to enable deterministic behavior.

**Call arguments**

- **inputs**: Input tensor (of any rank).
- **training**: Python boolean indicating whether the layer should behave in training mode (adding noise) or in inference mode (doing nothing).
