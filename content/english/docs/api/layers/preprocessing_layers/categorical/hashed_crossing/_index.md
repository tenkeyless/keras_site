---
title: HashedCrossing layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/hashed_crossing.py#L11" >}}

### `HashedCrossing` class

```python
keras.layers.HashedCrossing(
    num_bins, output_mode="int", sparse=False, name=None, dtype=None, **kwargs
)
```

A preprocessing layer which crosses features using the "hashing trick".

This layer performs crosses of categorical features using the "hashing trick". Conceptually, the transformation can be thought of as: `hash(concatenate(features)) % num_bins`.

This layer currently only performs crosses of scalar inputs and batches of scalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size,)` and `()`.

**Note:** This layer wraps [`tf.keras.layers.HashedCrossing`]({{< relref "/docs/api/layers/preprocessing_layers/categorical/hashed_crossing#hashedcrossing-class" >}}). It cannot be used as part of the compiled computation graph of a model with any backend other than TensorFlow. It can however be used with any backend when running eagerly. It can also always be used as part of an input preprocessing pipeline with any backend (outside the model itself), which is how we recommend to use this layer.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Arguments**

- **num_bins**: Number of hash bins.
- **output_mode**: Specification for the output of the layer. Values can be `"int"`, or `"one_hot"` configuring the layer as follows:
  - `"int"`: Return the integer bin indices directly.
  - `"one_hot"`: Encodes each individual element in the input into an array the same size as `num_bins`, containing a 1 at the input's bin index. Defaults to `"int"`.
- **sparse**: Boolean. Only applicable to `"one_hot"` mode and only valid when using the TensorFlow backend. If `True`, returns a `SparseTensor` instead of a dense `Tensor`. Defaults to `False`.
- **\*\*kwargs**: Keyword arguments to construct a layer.

**Examples**

**Crossing two scalar features.**

```console
>>> layer = keras.layers.HashedCrossing(
...     num_bins=5)
>>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])
>>> feat2 = np.array([101, 101, 101, 102, 102])
>>> layer((feat1, feat2))
array([1, 4, 1, 1, 3])
```

**Crossing and one-hotting two scalar features.**

```console
>>> layer = keras.layers.HashedCrossing(
...     num_bins=5, output_mode='one_hot')
>>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])
>>> feat2 = np.array([101, 101, 101, 102, 102])
>>> layer((feat1, feat2))
array([[0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0.]], dtype=float32)
```
