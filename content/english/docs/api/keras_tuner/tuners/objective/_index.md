---
title: Objective class
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/objective.py#L19" >}}

### `Objective` class

```python
keras_tuner.Objective(name, direction)
```

The objective for optimization during tuning.

**Arguments**

- **name**: String. The name of the objective.
- **direction**: String. The value should be "min" or "max" indicating
  whether the objective value should be minimized or maximized.
