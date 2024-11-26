---
title: RandAugment layer
toc: true
weight: 9
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/rand_augment.py#L24" >}}

### `RandAugment` class

```python
keras_cv.layers.RandAugment(
    value_range,
    augmentations_per_image=3,
    magnitude=0.5,
    magnitude_stddev=0.15,
    rate=0.9090909090909091,
    geometric=True,
    seed=None,
    **kwargs
)
```

RandAugment performs the Rand Augment operation on input images.

This layer can be thought of as an all-in-one image augmentation layer. The
policy implemented by this layer has been benchmarked extensively and is
effective on a wide variety of datasets.

The policy operates as follows:

For each augmentation in the range `[0, augmentations_per_image]`,
the policy selects a random operation from a list of operations.
It then samples a random number and if that number is less than
`rate` applies it to the given image.

**References**

- [RandAugment](https://arxiv.org/abs/1909.13719)

**Arguments**

- **value_range**: the range of values the incoming images will have.
  Represented as a two number tuple written [low, high].
  This is typically either `[0, 1]` or `[0, 255]` depending
  on how your preprocessing pipeline is set up.
- **augmentations_per_image**: the number of layers to use in the rand augment
  policy, defaults to `3`.
- **magnitude**: magnitude is the mean of the normal distribution used to
  sample the magnitude used for each data augmentation. Magnitude
  should be a float in the range `[0, 1]`. A magnitude of `0`
  indicates that the augmentations are as weak as possible (not
  recommended), while a value of `1.0` implies use of the strongest
  possible augmentation. All magnitudes are clipped to the range
  `[0, 1]` after sampling. Defaults to `0.5`.
- **magnitude_stddev**: the standard deviation to use when drawing values for
  the perturbations. Keep in mind magnitude will still be clipped to
  the range `[0, 1]` after samples are drawn from the normal
  distribution. Defaults to `0.15`.
- **rate**: the rate at which to apply each augmentation. This parameter is
  applied on a per-distortion layer, per image. Should be in the range
  `[0, 1]`. To reproduce the original RandAugment paper results, set
  this to `10/11`. The original `RandAugment` paper includes an
  Identity transform. By setting the rate to 10/11 in our
  implementation, the behavior is identical to sampling an Identity
  augmentation 10/11th of the time. Defaults to `1.0`.
- **geometric**: whether to include geometric augmentations. This
  should be set to False when performing object detection. Defaults to
  True.

**Example**

```python
(x_test, y_test), _ = keras.datasets.cifar10.load_data()
rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255), augmentations_per_image=3, magnitude=0.5
)
x_test = rand_augment(x_test)
```
