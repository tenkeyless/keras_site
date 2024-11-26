---
title: Normalization layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/normalization.py#L12" >}}

### `Normalization` class

```python
keras.layers.Normalization(axis=-1, mean=None, variance=None, invert=False, **kwargs)
```

A preprocessing layer that normalizes continuous features.

This layer will shift and scale inputs into a distribution centered around 0 with standard deviation 1. It accomplishes this by precomputing the mean and variance of the data, and calling `(input - mean) / sqrt(var)` at runtime.

The mean and variance values for the layer must be either supplied on construction or learned via `adapt()`. `adapt()` will compute the mean and variance of the data and store them as the layer's weights. `adapt()` should be called before `fit()`, `evaluate()`, or `predict()`.

**Arguments**

- **axis**: Integer, tuple of integers, or None. The axis or axes that should have a separate mean and variance for each index in the shape. For example, if shape is `(None, 5)` and `axis=1`, the layer will track 5 separate mean and variance values for the last axis. If `axis` is set to `None`, the layer will normalize all elements in the input by a scalar mean and variance. When `-1`, the last axis of the input is assumed to be a feature dimension and is normalized per index. Note that in the specific case of batched scalar inputs where the only axis is the batch axis, the default will normalize each index in the batch separately. In this case, consider passing `axis=None`. Defaults to `-1`.
- **mean**: The mean value(s) to use during normalization. The passed value(s) will be broadcast to the shape of the kept axes above; if the value(s) cannot be broadcast, an error will be raised when this layer's `build()` method is called.
- **variance**: The variance value(s) to use during normalization. The passed value(s) will be broadcast to the shape of the kept axes above; if the value(s) cannot be broadcast, an error will be raised when this layer's `build()` method is called.
- **invert**: If `True`, this layer will apply the inverse transformation to its inputs: it would turn a normalized input back into its original form.

**Examples**

Calculate a global mean and variance by analyzing the dataset in `adapt()`.

```console
>>> adapt_data = np.array([1., 2., 3., 4., 5.], dtype='float32')
>>> input_data = np.array([1., 2., 3.], dtype='float32')
>>> layer = keras.layers.Normalization(axis=None)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
array([-1.4142135, -0.70710677, 0.], dtype=float32)
```

Calculate a mean and variance for each index on the last axis.

```console
>>> adapt_data = np.array([[0., 7., 4.],
...                        [2., 9., 6.],
...                        [0., 7., 4.],
...                        [2., 9., 6.]], dtype='float32')
>>> input_data = np.array([[0., 7., 4.]], dtype='float32')
>>> layer = keras.layers.Normalization(axis=-1)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
array([-1., -1., -1.], dtype=float32)
```

Pass the mean and variance directly.

```console
>>> input_data = np.array([[1.], [2.], [3.]], dtype='float32')
>>> layer = keras.layers.Normalization(mean=3., variance=2.)
>>> layer(input_data)
array([[-1.4142135 ],
       [-0.70710677],
       [ 0.        ]], dtype=float32)
```

Use the layer to de-normalize inputs (after adapting the layer).

```console
>>> adapt_data = np.array([[0., 7., 4.],
...                        [2., 9., 6.],
...                        [0., 7., 4.],
...                        [2., 9., 6.]], dtype='float32')
>>> input_data = np.array([[1., 2., 3.]], dtype='float32')
>>> layer = keras.layers.Normalization(axis=-1, invert=True)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
array([2., 10., 8.], dtype=float32)
```
