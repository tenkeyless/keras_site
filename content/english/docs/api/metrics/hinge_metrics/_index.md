---
title: Hinge metrics for "maximum-margin" classification
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/hinge_metrics.py#L8" >}}

### `Hinge` class

```python
keras.metrics.Hinge(name="hinge", dtype=None)
```

Computes the hinge metric between `y_true` and `y_pred`.

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Examples**

```console
>>> m = keras.metrics.Hinge()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result()
1.3
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result()
1.1
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/hinge_metrics.py#L41" >}}

### `SquaredHinge` class

```python
keras.metrics.SquaredHinge(name="squared_hinge", dtype=None)
```

Computes the hinge metric between `y_true` and `y_pred`.

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.SquaredHinge()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result()
1.86
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result()
1.46
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/hinge_metrics.py#L74" >}}

### `CategoricalHinge` class

```python
keras.metrics.CategoricalHinge(name="categorical_hinge", dtype=None)
```

Computes the categorical hinge metric between `y_true` and `y_pred`.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.CategoricalHinge()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result().numpy()
1.4000001
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result()
1.2
```
