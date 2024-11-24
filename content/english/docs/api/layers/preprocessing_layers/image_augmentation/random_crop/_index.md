---
title: RandomCrop layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/image_preprocessing/random_crop.py#L12" >}}

### `RandomCrop` class

```python
keras.layers.RandomCrop(
    height, width, seed=None, data_format=None, name=None, **kwargs
)
```

A preprocessing layer which randomly crops images during training.

During training, this layer will randomly choose a location to crop images down to a target size. The layer will crop all the images in the same batch to the same cropping location.

At inference time, and during training if an input image is smaller than the target size, the input will be resized and cropped so as to return the largest possible window in the image that matches the target aspect ratio. If you need to apply random cropping at inference time, set `training` to True when calling the layer.

Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype. By default, the layer will output floats.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Input shape**

3D (unbatched) or 4D (batched) tensor with shape: `(..., height, width, channels)`, in `"channels_last"` format.

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape: `(..., target_height, target_width, channels)`.

**Arguments**

- **height**: Integer, the height of the output shape.
- **width**: Integer, the width of the output shape.
- **seed**: Integer. Used to create a random seed.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.
