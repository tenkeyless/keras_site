---
title: UnitNormalization layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/normalization/unit_normalization.py#L6" >}}

### `UnitNormalization` class

```python
keras.layers.UnitNormalization(axis=-1, **kwargs)
```

Unit normalization layer.

Normalize a batch of inputs so that each input in the batch has a L2 norm equal to 1 (across the axes specified in `axis`).

**Example**

```console
>>> data = np.arange(6).reshape(2, 3)
>>> normalized_data = keras.layers.UnitNormalization()(data)
>>> np.sum(normalized_data[0, :] ** 2)
1.0
```

**Arguments**

- **axis**: Integer or list/tuple. The axis or axes to normalize across. Typically, this is the features axis or axes. The left-out axes are typically the batch axis or axes. `-1` is the last dimension in the input. Defaults to `-1`.
