---
title: Focal Loss
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/losses/focal.py#L20" >}}

### `FocalLoss` class

```python
keras_cv.losses.FocalLoss(
    alpha=0.25, gamma=2, from_logits=False, label_smoothing=0, **kwargs
)
```

Implements Focal loss

Focal loss is a modified cross-entropy designed to perform better with
class imbalance. For this reason, it's commonly used with object detectors.

**Arguments**

- **alpha**: a float value between 0 and 1 representing a weighting factor
  used to deal with class imbalance. Positive classes and negative
  classes have alpha and (1 - alpha) as their weighting factors
  respectively. Defaults to 0.25.
- **gamma**: a positive float value representing the tunable focusing
  parameter, defaults to 2.
- **from_logits**: Whether `y_pred` is expected to be a logits tensor. By
  default, `y_pred` is assumed to encode a probability distribution.
  Default to `False`.
- **label_smoothing**: Float in `[0, 1]`. If higher than 0 then smooth the
  labels by squeezing them towards `0.5`, i.e., using
  `1. - 0.5 * label_smoothing` for the target class and
  `0.5 * label_smoothing` for the non-target class.

**References**

- [Focal Loss paper](https://arxiv.org/abs/1708.02002)

**Example**

```python
y_true = np.random.uniform(size=[10], low=0, high=4)
y_pred = np.random.uniform(size=[10], low=0, high=4)
loss = FocalLoss()
loss(y_true, y_pred)
```

Usage with the `compile()` API:

```python
model.compile(optimizer='adam', loss=keras_cv.losses.FocalLoss())
```
