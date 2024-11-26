---
title: Spectral Normalization layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/normalization/spectral_normalization.py#L9" >}}

### `SpectralNormalization` class

```python
keras.layers.SpectralNormalization(layer, power_iterations=1, **kwargs)
```

Performs spectral normalization on the weights of a target layer.

This wrapper controls the Lipschitz constant of the weights of a layer by constraining their spectral norm, which can stabilize the training of GANs.

**Arguments**

- **layer**: A [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) instance that has either a `kernel` (e.g. `Conv2D`, `Dense`...) or an `embeddings` attribute (`Embedding` layer).
- **power_iterations**: int, the number of iterations during normalization.
- **\*\*kwargs**: Base wrapper keyword arguments.

**Examples**

Wrap [`keras.layers.Conv2D`]({{< relref "/docs/api/layers/convolution_layers/convolution2d#conv2d-class" >}}):

```console
>>> x = np.random.rand(1, 10, 10, 1)
>>> conv2d = SpectralNormalization(keras.layers.Conv2D(2, 2))
>>> y = conv2d(x)
>>> y.shape
(1, 9, 9, 2)
```

Wrap [`keras.layers.Dense`]({{< relref "/docs/api/layers/core_layers/dense#dense-class" >}}):

```console
>>> x = np.random.rand(1, 10, 10, 1)
>>> dense = SpectralNormalization(keras.layers.Dense(10))
>>> y = dense(x)
>>> y.shape
(1, 10, 10, 10)
```

**Reference**

- [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).
