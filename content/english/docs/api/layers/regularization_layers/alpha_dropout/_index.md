---
title: AlphaDropout layer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/regularization/alpha_dropout.py#L7" >}}

### `AlphaDropout` class

```python
keras.layers.AlphaDropout(rate, noise_shape=None, seed=None, **kwargs)
```

Applies Alpha Dropout to the input.

Alpha Dropout is a `Dropout` that keeps mean and variance of inputs to their original values, in order to ensure the self-normalizing property even after this dropout. Alpha Dropout fits well to Scaled Exponential Linear Units (SELU) by randomly setting activations to the negative saturation value.

**Arguments**

- **rate**: Float between 0 and 1. The multiplicative noise will have standard deviation `sqrt(rate / (1 - rate))`.
- **noise_shape**: 1D integer tensor representing the shape of the binary alpha dropout mask that will be multiplied with the input. For instance, if your inputs have shape `(batch_size, timesteps, features)` and you want the alpha dropout mask to be the same for all timesteps, you can use `noise_shape=(batch_size, 1, features)`.
- **seed**: A Python integer to use as random seed.

**Call arguments**

- **inputs**: Input tensor (of any rank).
- **training**: Python boolean indicating whether the layer should behave in training mode (adding alpha dropout) or in inference mode (doing nothing).
