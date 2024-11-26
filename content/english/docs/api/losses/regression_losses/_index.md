---
title: Regression losses
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L42" >}}

### `MeanSquaredError` class

```python
keras.losses.MeanSquaredError(
    reduction="sum_over_batch_size", name="mean_squared_error", dtype=None
)
```

Computes the mean of squares of errors between labels and predictions.

Formula:

```python
loss = mean(square(y_true - y_pred))
```

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

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L78" >}}

### `MeanAbsoluteError` class

```python
keras.losses.MeanAbsoluteError(
    reduction="sum_over_batch_size", name="mean_absolute_error", dtype=None
)
```

Computes the mean of absolute difference between labels and predictions.

Formula:

```python
loss = mean(abs(y_true - y_pred))
```

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

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L114" >}}

### `MeanAbsolutePercentageError` class

```python
keras.losses.MeanAbsolutePercentageError(
    reduction="sum_over_batch_size", name="mean_absolute_percentage_error", dtype=None
)
```

Computes the mean absolute percentage error between `y_true` & `y_pred`.

Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true))
```

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

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L153" >}}

### `MeanSquaredLogarithmicError` class

```python
keras.losses.MeanSquaredLogarithmicError(
    reduction="sum_over_batch_size", name="mean_squared_logarithmic_error", dtype=None
)
```

Computes the mean squared logarithmic error between `y_true` & `y_pred`.

Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
```

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

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L192" >}}

### `CosineSimilarity` class

```python
keras.losses.CosineSimilarity(
    axis=-1, reduction="sum_over_batch_size", name="cosine_similarity", dtype=None
)
```

Computes the cosine similarity between `y_true` & `y_pred`.

Note that it is a number between -1 and 1. When it is a negative number
between -1 and 0, 0 indicates orthogonality and values closer to -1
indicate greater similarity. This makes it usable as a loss function in a
setting where you try to maximize the proximity between predictions and
targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity
will be 0 regardless of the proximity between predictions and targets.

Formula:

```python
loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
```

**Arguments**

- **axis**: The axis along which the cosine similarity is computed
  (the features axis). Defaults to `-1`.
- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L242" >}}

### `Huber` class

```python
keras.losses.Huber(
    delta=1.0, reduction="sum_over_batch_size", name="huber_loss", dtype=None
)
```

Computes the Huber loss between `y_true` & `y_pred`.

Formula:

```python
for x in error:
    if abs(x) <= delta:
        loss.append(0.5 * x^2)
    elif abs(x) > delta:
        loss.append(delta * abs(x) - 0.5 * delta^2)
loss = mean(loss, axis=-1)
```

See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

**Arguments**

- **delta**: A float, the point where the Huber loss function changes from a
  quadratic to linear.
- **reduction**: Type of reduction to apply to loss. Options are `"sum"`,
  `"sum_over_batch_size"` or `None`. Defaults to
  `"sum_over_batch_size"`.
- **name**: Optional name for the instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L288" >}}

### `LogCosh` class

```python
keras.losses.LogCosh(reduction="sum_over_batch_size", name="log_cosh", dtype=None)
```

Computes the logarithm of the hyperbolic cosine of the prediction error.

Formula:

```python
error = y_pred - y_true
logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)`
```

where x is the error `y_pred - y_true`.

**Arguments**

- **reduction**: Type of reduction to apply to loss. Options are `"sum"`,
  `"sum_over_batch_size"` or `None`. Defaults to
  `"sum_over_batch_size"`.
- **name**: Optional name for the instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L2202" >}}

### `Tversky` class

```python
keras.losses.Tversky(
    alpha=0.5, beta=0.5, reduction="sum_over_batch_size", name="tversky", dtype=None
)
```

Computes the Tversky loss value between `y_true` and `y_pred`.

This loss function is weighted by the alpha and beta coefficients
that penalize false positives and false negatives.

With `alpha=0.5` and `beta=0.5`, the loss value becomes equivalent to
Dice Loss.

**Arguments**

- **alpha**: The coefficient controlling incidence of false positives.
  Defaults to `0.5`.
- **beta**: The coefficient controlling incidence of false negatives.
  Defaults to `0.5`.
- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

**Returns**

Tversky loss value.

**Reference**

- [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L2093" >}}

### `Dice` class

```python
keras.losses.Dice(
    reduction="sum_over_batch_size", name="dice", axis=None, dtype=None
)
```

Computes the Dice loss value between `y_true` and `y_pred`.

Formula:

```python
loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
```

**Arguments**

- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **axis**: Tuple for which dimensions the loss is calculated. Defaults to
  `None`.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

**Returns**

Dice loss value.

**Example**

```console
>>> y_true = [[[[1.0], [1.0]], [[0.0], [0.0]]],
...           [[[1.0], [1.0]], [[0.0], [0.0]]]]
>>> y_pred = [[[[0.0], [1.0]], [[0.0], [1.0]]],
...           [[[0.4], [0.0]], [[0.0], [0.9]]]]
>>> axis = (1, 2, 3)
>>> loss = keras.losses.dice(y_true, y_pred, axis=axis)
>>> assert loss.shape == (2,)
>>> loss
array([0.5, 0.75757575], shape=(2,), dtype=float32)
```

```console
>>> loss = keras.losses.dice(y_true, y_pred)
>>> assert loss.shape == ()
>>> loss
array(0.6164384, shape=(), dtype=float32)
```

```console
>>> y_true = np.array(y_true)
>>> y_pred = np.array(y_pred)
>>> loss = keras.losses.Dice(axis=axis, reduction=None)(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss
array([0.5, 0.75757575], shape=(2,), dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1267" >}}

### `mean_squared_error` function

```python
keras.losses.mean_squared_error(y_true, y_pred)
```

Computes the mean squared error between labels and predictions.

Formula:

```python
loss = mean(square(y_true - y_pred), axis=-1)
```

**Example**

```console
>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_squared_error(y_true, y_pred)
```

**Arguments**

- **y_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1306" >}}

### `mean_absolute_error` function

```python
keras.losses.mean_absolute_error(y_true, y_pred)
```

Computes the mean absolute error between labels and predictions.

```python
loss = mean(abs(y_true - y_pred), axis=-1)
```

**Arguments**

- **y_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```console
>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_absolute_error(y_true, y_pred)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1343" >}}

### `mean_absolute_percentage_error` function

```python
keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```

Computes the mean absolute percentage error between `y_true` & `y_pred`.

Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
```

Division by zero is prevented by dividing by `maximum(y_true, epsilon)`
where `epsilon = keras.backend.epsilon()`
(default to `1e-7`).

**Arguments**

- **y_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean absolute percentage error values with shape = `[batch_size, d0, ..
dN-1]`.

**Example**

```console
>>> y_true = np.random.random(size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1389" >}}

### `mean_squared_logarithmic_error` function

```python
keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

Computes the mean squared logarithmic error between `y_true` & `y_pred`.

Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
```

Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
values and 0 values will be replaced with `keras.backend.epsilon()`
(default to `1e-7`).

**Arguments**

- **y_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean squared logarithmic error values with shape = `[batch_size, d0, ..
dN-1]`.

**Example**

```console
>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1436" >}}

### `cosine_similarity` function

```python
keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
```

Computes the cosine similarity between labels and predictions.

Formula:

```python
loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
```

Note that it is a number between -1 and 1. When it is a negative number
between -1 and 0, 0 indicates orthogonality and values closer to -1
indicate greater similarity. This makes it usable as a loss function in a
setting where you try to maximize the proximity between predictions and
targets. If either `y_true` or `y_pred` is a zero vector, cosine
similarity will be 0 regardless of the proximity between predictions
and targets.

**Arguments**

- **y_true**: Tensor of true targets.
- **y_pred**: Tensor of predicted targets.
- **axis**: Axis along which to determine similarity. Defaults to `-1`.

**Returns**

Cosine similarity tensor.

**Example**

```console
>>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
>>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
>>> loss = keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
[-0., -0.99999994, 0.99999994]
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1476" >}}

### `huber` function

```python
keras.losses.huber(y_true, y_pred, delta=1.0)
```

Computes Huber loss value.

Formula:

```python
for x in error:
    if abs(x) <= delta:
        loss.append(0.5 * x^2)
    elif abs(x) > delta:
        loss.append(delta * abs(x) - 0.5 * delta^2)
loss = mean(loss, axis=-1)
```

See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

**Example**

```console
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = keras.losses.huber(y_true, y_pred)
0.155
```

**Arguments**

- **y_true**: tensor of true targets.
- **y_pred**: tensor of predicted targets.
- **delta**: A float, the point where the Huber loss function changes from a
  quadratic to linear. Defaults to `1.0`.

**Returns**

Tensor with one scalar loss entry per sample.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1526" >}}

### `log_cosh` function

```python
keras.losses.log_cosh(y_true, y_pred)
```

Logarithm of the hyperbolic cosine of the prediction error.

Formula:

```python
loss = mean(log(cosh(y_pred - y_true)), axis=-1)
```

Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small
`x` and to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works
mostly like the mean squared error, but will not be so strongly affected by
the occasional wildly incorrect prediction.

**Example**

```console
>>> y_true = [[0., 1.], [0., 0.]]
>>> y_pred = [[1., 1.], [0., 0.]]
>>> loss = keras.losses.log_cosh(y_true, y_pred)
0.108
```

**Arguments**

- **y_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L2263" >}}

### `tversky` function

```python
keras.losses.tversky(y_true, y_pred, alpha=0.5, beta=0.5)
```

Computes the Tversky loss value between `y_true` and `y_pred`.

This loss function is weighted by the alpha and beta coefficients
that penalize false positives and false negatives.

With `alpha=0.5` and `beta=0.5`, the loss value becomes equivalent to
Dice Loss.

**Arguments**

- **y_true**: tensor of true targets.
- **y_pred**: tensor of predicted targets.
- **alpha**: coefficient controlling incidence of false positives.
- **beta**: coefficient controlling incidence of false negatives.

**Returns**

Tversky loss value.

**Reference**

- [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L2168" >}}

### `dice` function

```python
keras.losses.dice(y_true, y_pred, axis=None)
```

Computes the Dice loss value between `y_true` and `y_pred`.

Formula:

```python
loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
```

**Arguments**

- **y_true**: tensor of true targets.
- **y_pred**: tensor of predicted targets.
- **axis**: tuple for which dimensions the loss is calculated

**Returns**

Dice loss value.
