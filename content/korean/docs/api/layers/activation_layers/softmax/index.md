---
title: Softmax layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/activations/softmax.py#L14" >}}

### `Softmax` class

```python
keras.layers.Softmax(axis=-1, **kwargs)
```

Softmax activation layer.

Formula:

```python
exp_x = exp(x - max(x))
f(x) = exp_x / sum(exp_x)
```

**Example**

```console
>>>softmax_layer = keras.layers.activations.Softmax()
>>>input = np.array([1.0, 2.0, 1.0])
>>>result = softmax_layer(input)
[0.21194157, 0.5761169, 0.21194157]
```

**Arguments**

- **axis**: Integer, or list of Integers, axis along which the softmax
  normalization is applied.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.

**Call arguments**

- **inputs**: The inputs (logits) to the softmax layer.
- **mask**: A boolean mask of the same shape as `inputs`. The mask
  specifies 1 to keep and 0 to mask. Defaults to `None`.

**Returns**

Softmaxed output with the same shape as `inputs`.
