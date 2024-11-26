---
title: DropPath layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/regularization/drop_path.py#L21" >}}

### `DropPath` class

```python
keras_cv.layers.DropPath(rate=0.5, seed=None, **kwargs)
```

Implements the DropPath layer. DropPath randomly drops samples during
training with a probability of `rate`. Note that this layer drops individual
samples within a batch and not the entire batch. DropPath randomly drops
some individual samples from a batch, whereas StochasticDepth
randomly drops the entire batch.

**References**

- [FractalNet](https://arxiv.org/abs/1605.07648v4).
  - [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/layers/drop.py#L135)

**Arguments**

- **rate**: float, the probability of the residual branch being dropped.
- **seed**: (Optional) integer. Used to create a random seed.

**Example**

`DropPath` can be used in any network as follows:

```python
# (...)
input = tf.ones((1, 3, 3, 1), dtype=tf.float32)
residual = keras.layers.Conv2D(1, 1)(input)
output = keras_cv.layers.DropPath()(input)
# (...)
```
