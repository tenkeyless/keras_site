---
title: ChannelShuffle layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/channel_shuffle.py#L23" >}}

### `ChannelShuffle` class

```python
keras_cv.layers.ChannelShuffle(groups=3, seed=None, **kwargs)
```

Shuffle channels of an input image.

**Input shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format

**Arguments**

- **groups**: Number of groups to divide the input channels, defaults to 3.
- **seed**: Integer. Used to create a random seed.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
channel_shuffle = ChannelShuffle(groups=3)
augmented_images = channel_shuffle(images)
```
