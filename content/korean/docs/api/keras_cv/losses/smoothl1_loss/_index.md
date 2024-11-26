---
title: SmoothL1Loss Loss
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/losses/smooth_l1.py#L20" >}}

### `SmoothL1Loss` class

```python
keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, **kwargs)
```

Implements Smooth L1 loss.

SmoothL1Loss implements the SmoothL1 function, where values less than
`l1_cutoff` contribute to the overall loss based on their squared
difference, and values greater than l1_cutoff contribute based on their raw
difference.

**Arguments**

- **l1_cutoff**: differences between y_true and y_pred that are larger than
  `l1_cutoff` are treated as `L1` values
