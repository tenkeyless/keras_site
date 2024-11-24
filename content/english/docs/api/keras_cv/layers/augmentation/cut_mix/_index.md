---
title: cut_mix
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/cut_mix.py#L24" >}}

### `CutMix` class

`keras_cv.layers.CutMix(alpha=1.0, seed=None, **kwargs)`

CutMix implements the CutMix data augmentation technique.

**Arguments**

- **alpha**: Float between 0 and 1. Inverse scale parameter for the gamma distribution. This controls the shape of the distribution from which the smoothing values are sampled. Defaults to 1.0, which is a recommended value when training an imagenet1k classification model.
- **seed**: Integer. Used to create a random seed.

**References**

- [CutMix paper](https://arxiv.org/abs/1905.04899).

---
