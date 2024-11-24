---
title: Rescaling layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/rescaling.py#L7" >}}

### `Rescaling` class

```python
keras.layers.Rescaling(scale, offset=0.0, **kwargs)
```

A preprocessing layer which rescales input values to a new range.

This layer rescales every value of an input (often an image) by multiplying by `scale` and adding `offset`.

For instance:

1.  To rescale an input in the `[0, 255]` range to be in the `[0, 1]` range, you would pass `scale=1./255`.
2.  To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range, you would pass `scale=1./127.5, offset=-1`.

The rescaling is applied both during training and inference. Inputs can be of integer or floating point dtype, and by default the layer will output floats.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Arguments**

- **scale**: Float, the scale to apply to the inputs.
- **offset**: Float, the offset to apply to the inputs.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.
