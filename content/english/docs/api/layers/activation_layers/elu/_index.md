---
title: ELU layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/activations/elu.py#L6" >}}

### `ELU` class

```python
keras.layers.ELU(alpha=1.0, **kwargs)
```

Applies an Exponential Linear Unit function to an output.

Formula:

```python
f(x) = alpha * (exp(x) - 1.) for x < 0
f(x) = x for x >= 0
```

**Arguments**

- **alpha**: float, slope of negative section. Defaults to `1.0`.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.
