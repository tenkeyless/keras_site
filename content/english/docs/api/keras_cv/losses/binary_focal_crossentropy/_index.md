---
title: Binary Penalty Reduced Focal CrossEntropy
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/losses/penalty_reduced_focal_loss.py#L20" >}}

### `BinaryPenaltyReducedFocalCrossEntropy` class

```python
keras_cv.losses.BinaryPenaltyReducedFocalCrossEntropy(
    alpha=2.0,
    beta=4.0,
    from_logits=False,
    positive_threshold=0.99,
    positive_weight=1.0,
    negative_weight=1.0,
    reduction="sum_over_batch_size",
    name="binary_penalty_reduced_focal_cross_entropy",
)
```

Implements CenterNet modified Focal loss.

Compared with [`keras.losses.BinaryFocalCrossentropy`]({{< relref "/docs/api/losses/probabilistic_losses#binaryfocalcrossentropy-class" >}}), this loss discounts
for negative labels that have value less than `positive_threshold`, the
larger value the negative label is, the more discount to the final loss.

User can choose to divide the number of keypoints outside the loss
computation, or by passing in `sample_weight` as 1.0/num_key_points.

**Arguments**

- **alpha**: a focusing parameter used to compute the focal factor.
  Defaults to 2.0. Note, this is equivalent to the `gamma` parameter in
  [`keras.losses.BinaryFocalCrossentropy`]({{< relref "/docs/api/losses/probabilistic_losses#binaryfocalcrossentropy-class" >}}).
- **beta**: a float parameter, penalty exponent for negative labels, defaults to
  4.0.
- **from_logits**: Whether `y_pred` is expected to be a logits tensor, defaults
  to `False`.
- **positive_threshold**: Anything bigger than this is treated as positive
  label, defaults to 0.99.
- **positive_weight**: single scalar weight on positive examples, defaults to
  1.0.
- **negative_weight**: single scalar weight on negative examples, defaults to
  1.0.

Inputs:
y_true: [batch\_size, ...] float tensor
y_pred: [batch\_size, ...] float tensor with same shape as y_true.

**References**

- [Objects as Points](https://arxiv.org/pdf/1904.07850.pdf) Eq 1.
  - [Cornernet: Detecting objects as paired keypoints](https://arxiv.org/abs/1808.01244) for `alpha` and
    `beta`.
