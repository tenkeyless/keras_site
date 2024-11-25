---
title: SimCLR Loss
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/losses/simclr_loss.py#L29" >}}

### `SimCLRLoss` class

```python
keras_cv.losses.SimCLRLoss(temperature, **kwargs)
```

Implements SimCLR Cosine Similarity loss.

SimCLR loss is used for contrastive self-supervised learning.

**Arguments**

- **temperature**: a float value between 0 and 1, used as a scaling factor for
  cosine similarity.

**References**

- [SimCLR paper](https://arxiv.org/pdf/2002.05709)
