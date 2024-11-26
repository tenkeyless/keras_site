---
title: Permute layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/reshaping/permute.py#L8" >}}

### `Permute` class

```python
keras.layers.Permute(dims, **kwargs)
```

Permutes the dimensions of the input according to a given pattern.

Useful e.g. connecting RNNs and convnets.

**Arguments**

- **dims**: Tuple of integers. Permutation pattern does not include the batch dimension. Indexing starts at 1. For instance, `(1, 3, 2)` permutes the second and third dimensions of the input.

**Input shape**

Arbitrary.

**Output shape**

Same as the input shape, but with the dimensions re-ordered according to the specified pattern.

**Example**

```console
>>> x = keras.Input(shape=(10, 64))
>>> y = keras.layers.Permute((2, 1))(x)
>>> y.shape
(None, 64, 10)
```
