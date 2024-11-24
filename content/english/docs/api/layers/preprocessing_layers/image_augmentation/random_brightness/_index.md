---
title: RandomBrightness layer
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/image_preprocessing/random_brightness.py#L8" >}}

### `RandomBrightness` class

```python
keras.layers.RandomBrightness(factor, value_range=(0, 255), seed=None, **kwargs)
```

A preprocessing layer which randomly adjusts brightness during training.

This layer will randomly increase/reduce the brightness for the input RGB images. At inference time, the output will be identical to the input. Call the layer with `training=True` to adjust the brightness of the input.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Arguments**

- **factor**: Float or a list/tuple of 2 floats between -1.0 and 1.0. The factor is used to determine the lower bound and upper bound of the brightness adjustment. A float value will be chosen randomly between the limits. When -1.0 is chosen, the output image will be black, and when 1.0 is chosen, the image will be fully white. When only one float is provided, eg, 0.2, then -0.2 will be used for lower bound and 0.2 will be used for upper bound.
- **value_range**: Optional list/tuple of 2 floats for the lower and upper limit of the values of the input data. To make no change, use `[0.0, 1.0]`, e.g., if the image input has been scaled before this layer. Defaults to `[0.0, 255.0]`. The brightness adjustment will be scaled to this range, and the output values will be clipped to this range.
- **seed**: optional integer, for fixed RNG behavior.

Inputs: 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)

Output: 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the `factor`. By default, the layer will output floats. The output value will be clipped to the range `[0, 255]`, the valid range of RGB colors, and rescaled based on the `value_range` if needed.

**Example**

````console
random_bright = keras.layers.RandomBrightness(factor=0.2)

# An image with shape [2, 2, 3]
image = [[[1, 2, 3], [4 ,5 ,6]], [[7, 8, 9], [10, 11, 12]]]

# Assume we randomly select the factor to be 0.1, then it will apply
# 0.1 * 255 to all the channel
output = random_bright(image, training=True)

# output will be int64 with 25.5 added to each channel and round down.
```python
>>> array([[[26.5, 27.5, 28.5]
            [29.5, 30.5, 31.5]]
           [[32.5, 33.5, 34.5]
            [35.5, 36.5, 37.5]]],
          shape=(2, 2, 3), dtype=int64)
````
