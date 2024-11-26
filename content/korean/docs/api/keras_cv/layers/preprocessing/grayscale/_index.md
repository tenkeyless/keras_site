---
title: Grayscale layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/grayscale.py#L23" >}}

### `Grayscale` class

```python
keras_cv.layers.Grayscale(output_channels=1, **kwargs)
```

Grayscale is a preprocessing layer that transforms RGB images to
Grayscale images.
Input images should have values in the range of [0, 255].

**Input shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format

**Arguments**

output_channels.
Number color channels present in the output image.
The output_channels can be 1 or 3. RGB image with shape
(..., height, width, 3) will have the following shapes
after the `Grayscale` operation:
a. (..., height, width, 1) if output_channels = 1
b. (..., height, width, 3) if output_channels = 3.

**Example**

```python
(images, labels), _ = keras.datasets.cifar10.load_data()
to_grayscale = keras_cv.layers.preprocessing.Grayscale()
augmented_images = to_grayscale(images)
```
