---
title: FourierMix layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/fourier_mix.py#L23" >}}

### `FourierMix` class

```python
keras_cv.layers.FourierMix(alpha=0.5, decay_power=3, seed=None, **kwargs)
```

FourierMix implements the FMix data augmentation technique.

**Arguments**

- **alpha**: Float value for beta distribution. Inverse scale parameter for
  the gamma distribution. This controls the shape of the distribution
  from which the smoothing values are sampled. Defaults to 0.5, which
  is a recommended value in the paper.
- **decay_power**: A float value representing the decay power, defaults to 3,
  as recommended in the paper.
- **seed**: Integer. Used to create a random seed.

**References**

- [FMix paper](https://arxiv.org/abs/2002.12047).

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
fourier_mix = keras_cv.layers.preprocessing.FourierMix(0.5)
augmented_images, updated_labels = fourier_mix(
    {'images': images, 'labels': labels}
)
# output == {'images': updated_images, 'labels': updated_labels}
```
