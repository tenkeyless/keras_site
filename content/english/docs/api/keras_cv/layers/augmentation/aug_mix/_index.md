---
title: AugMix layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/aug_mix.py#L25" >}}

### `AugMix` class

```python
keras_cv.layers.AugMix(
    value_range,
    severity=0.3,
    num_chains=3,
    chain_depth=[1, 3],
    alpha=1.0,
    seed=None,
    **kwargs
)
```

Performs the AugMix data augmentation technique.

AugMix aims to produce images with variety while preserving the image
semantics and local statistics. During the augmentation process, each image
is augmented `num_chains` different ways, each way consisting of
`chain_depth` augmentations. Augmentations are sampled from the list:
translation, shearing, rotation, posterization, histogram equalization,
solarization and auto contrast. The results of each chain are then mixed
together with the original image based on random samples from a Dirichlet
distribution.

**Arguments**

- **value_range**: the range of values the incoming images will have.
  Represented as a two number tuple written (low, high).
  This is typically either `(0, 1)` or `(0, 255)` depending
  on how your preprocessing pipeline is set up.
- **severity**: A tuple of two floats, a single float or a
  `keras_cv.FactorSampler`. A value is sampled from the provided
  range. If a float is passed, the range is interpreted as
  `(0, severity)`. This value represents the level of strength of
  augmentations and is in the range [0, 1]. Defaults to 0.3.
- **num_chains**: an integer representing the number of different chains to
  be mixed, defaults to 3.
- **chain_depth**: an integer or range representing the number of
  transformations in the chains. If a range is passed, a random
  `chain_depth` value sampled from a uniform distribution over the
  given range is called at the start of the chain. Defaults to [1,3].
- **alpha**: a float value used as the probability coefficients for the
  Beta and Dirichlet distributions, defaults to 1.0.
- **seed**: Integer. Used to create a random seed.

**References**

- [AugMix paper](https://arxiv.org/pdf/1912.02781)
  - [Official Code](https://github.com/google-research/augmix)
  - [Unofficial TF Code](https://github.com/szacho/augmix-tf)

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
augmix = keras_cv.layers.AugMix([0, 255])
augmented_images = augmix(images[:100])
```
