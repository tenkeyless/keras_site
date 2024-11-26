---
title: Hinge losses for "maximum-margin" classification
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L321" >}}

### `Hinge` class

```python
keras.losses.Hinge(reduction="sum_over_batch_size", name="hinge", dtype=None)
```

Computes the hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = maximum(1 - y_true * y_pred, 0)
```

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

**Arguments**

- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L355" >}}

### `SquaredHinge` class

```python
keras.losses.SquaredHinge(
    reduction="sum_over_batch_size", name="squared_hinge", dtype=None
)
```

Computes the squared hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = square(maximum(1 - y_true * y_pred, 0))
```

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

**Arguments**

- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L391" >}}

### `CategoricalHinge` class

```python
keras.losses.CategoricalHinge(
    reduction="sum_over_batch_size", name="categorical_hinge", dtype=None
)
```

Computes the categorical hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = maximum(neg - pos + 1, 0)
```

where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

**Arguments**

- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1151" >}}

### `hinge` function

```python
keras.losses.hinge(y_true, y_pred)
```

Computes the hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)
```

**Arguments**

- **y_true**: The ground truth values. `y_true` values are expected to be -1
  or 1. If binary (0 or 1) labels are provided they will be converted
  to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```console
>>> y_true = np.random.choice([-1, 1], size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.hinge(y_true, y_pred)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1188" >}}

### `squared_hinge` function

```python
keras.losses.squared_hinge(y_true, y_pred)
```

Computes the squared hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)
```

**Arguments**

- **y_true**: The ground truth values. `y_true` values are expected to be -1
  or 1. If binary (0 or 1) labels are provided we will convert them
  to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Squared hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```console
>>> y_true = np.random.choice([-1, 1], size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.squared_hinge(y_true, y_pred)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1226" >}}

### `categorical_hinge` function

```python
keras.losses.categorical_hinge(y_true, y_pred)
```

Computes the categorical hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = maximum(neg - pos + 1, 0)
```

where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

**Arguments**

- **y_true**: The ground truth values. `y_true` values are expected to be
  either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
  shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```console
>>> y_true = np.random.randint(0, 3, size=(2,))
>>> y_true = np.eye(np.max(y_true) + 1)[y_true]
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.categorical_hinge(y_true, y_pred)
```
