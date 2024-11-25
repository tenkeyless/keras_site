---
title: GaussianDropout layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/regularization/gaussian_dropout.py#L9" >}}

### `GaussianDropout` class

```python
keras.layers.GaussianDropout(rate, seed=None, **kwargs)
```

Apply multiplicative 1-centered Gaussian noise.

As it is a regularization layer, it is only active at training time.

**Arguments**

- **rate**: Float, drop probability (as with `Dropout`). The multiplicative noise will have standard deviation `sqrt(rate / (1 - rate))`.
- **seed**: Integer, optional random seed to enable deterministic behavior.

**Call arguments**

- **inputs**: Input tensor (of any rank).
- **training**: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing).
