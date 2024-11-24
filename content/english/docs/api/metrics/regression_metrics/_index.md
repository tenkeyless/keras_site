---
title: Regression metrics
toc: true
weight: 4
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L16" >}}

### `MeanSquaredError` class

`keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)`

Computes the mean squared error between `y_true` and `y_pred`.

Formula:

`loss = mean(square(y_true - y_pred))`

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

`>>> m = keras.metrics.MeanSquaredError() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]]) >>> m.result() 0.25`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L186" >}}

### `RootMeanSquaredError` class

`keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)`

Computes root mean squared error metric between `y_true` and `y_pred`.

Formula:

`loss = sqrt(mean((y_pred - y_true) ** 2))`

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

**Example**

`>>> m = keras.metrics.RootMeanSquaredError() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]]) >>> m.result() 0.5`

`>>> m.reset_state() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]], ...                sample_weight=[1, 0]) >>> m.result() 0.70710677`

Usage with `compile()` API:

`model.compile(     optimizer='sgd',     loss='mse',     metrics=[keras.metrics.RootMeanSquaredError()])`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L47" >}}

### `MeanAbsoluteError` class

`keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)`

Computes the mean absolute error between the labels and predictions.

Formula:

`loss = mean(abs(y_true - y_pred))`

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Examples**

`>>> m = keras.metrics.MeanAbsoluteError() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]]) >>> m.result() 0.25 >>> m.reset_state() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]], ...                sample_weight=[1, 0]) >>> m.result() 0.5`

Usage with `compile()` API:

`model.compile(     optimizer='sgd',     loss='mse',     metrics=[keras.metrics.MeanAbsoluteError()])`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L92" >}}

### `MeanAbsolutePercentageError` class

`keras.metrics.MeanAbsolutePercentageError(     name="mean_absolute_percentage_error", dtype=None )`

Computes mean absolute percentage error between `y_true` and `y_pred`.

Formula:

`loss = 100 * mean(abs((y_true - y_pred) / y_true))`

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

**Example**

`>>> m = keras.metrics.MeanAbsolutePercentageError() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]]) >>> m.result() 250000000.0 >>> m.reset_state() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]], ...                sample_weight=[1, 0]) >>> m.result() 500000000.0`

Usage with `compile()` API:

`model.compile(     optimizer='sgd',     loss='mse',     metrics=[keras.metrics.MeanAbsolutePercentageError()])`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L139" >}}

### `MeanSquaredLogarithmicError` class

`keras.metrics.MeanSquaredLogarithmicError(     name="mean_squared_logarithmic_error", dtype=None )`

Computes mean squared logarithmic error between `y_true` and `y_pred`.

Formula:

`loss = mean(square(log(y_true + 1) - log(y_pred + 1)))`

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

**Example**

`>>> m = keras.metrics.MeanSquaredLogarithmicError() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]]) >>> m.result() 0.12011322 >>> m.reset_state() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]], ...                sample_weight=[1, 0]) >>> m.result() 0.24022643`

Usage with `compile()` API:

`model.compile(     optimizer='sgd',     loss='mse',     metrics=[keras.metrics.MeanSquaredLogarithmicError()])`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L254" >}}

### `CosineSimilarity` class

`keras.metrics.CosineSimilarity(name="cosine_similarity", dtype=None, axis=-1)`

Computes the cosine similarity between the labels and predictions.

Formula:

`loss = sum(l2_norm(y_true) * l2_norm(y_pred))`

See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity). This metric keeps the average cosine similarity between `predictions` and `labels` over a stream of data.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.
- **axis**: (Optional) Defaults to `-1`. The dimension along which the cosine similarity is computed.

**Example**

**Example**

`>>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]] >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]] >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]] >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1)) >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2 >>> m = keras.metrics.CosineSimilarity(axis=1) >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]]) >>> m.result() 0.49999997 >>> m.reset_state() >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]], ...                sample_weight=[0.3, 0.7]) >>> m.result() 0.6999999`

Usage with `compile()` API:

`model.compile(     optimizer='sgd',     loss='mse',     metrics=[keras.metrics.CosineSimilarity(axis=1)])`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L311" >}}

### `LogCoshError` class

`keras.metrics.LogCoshError(name="logcosh", dtype=None)`

Computes the logarithm of the hyperbolic cosine of the prediction error.

Formula:

`error = y_pred - y_true logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)`

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

**Example**

`>>> m = keras.metrics.LogCoshError() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]]) >>> m.result() 0.10844523 >>> m.reset_state() >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]], ...                sample_weight=[1, 0]) >>> m.result() 0.21689045`

Usage with `compile()` API:

`model.compile(optimizer='sgd',               loss='mse',               metrics=[keras.metrics.LogCoshError()])`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/regression_metrics.py#L359" >}}

### `R2Score` class

`keras.metrics.R2Score(     class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None )`

Computes R2 score.

Formula:

`sum_squares_residuals = sum((y_true - y_pred) ** 2) sum_squares = sum((y_true - mean(y_true)) ** 2) R2 = 1 - sum_squares_residuals / sum_squares`

This is also called the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).

It indicates how close the fitted regression line is to ground-truth data.

- The highest score possible is 1.0. It indicates that the predictors perfectly accounts for variation in the target.
- A score of 0.0 indicates that the predictors do not account for variation in the target.
- It can also be negative if the model is worse than random.

This metric can also compute the "Adjusted R2" score.

**Arguments**

- **class_aggregation**: Specifies how to aggregate scores corresponding to different output classes (or target dimensions), i.e. different dimensions on the last axis of the predictions. Equivalent to `multioutput` argument in Scikit-Learn. Should be one of `None` (no aggregation), `"uniform_average"`, `"variance_weighted_average"`.
- **num_regressors**: Number of independent regressors used ("Adjusted R2" score). 0 is the standard R2 score. Defaults to `0`.
- **name**: Optional. string name of the metric instance.
- **dtype**: Optional. data type of the metric result.

**Example**

`>>> y_true = np.array([[1], [4], [3]], dtype=np.float32) >>> y_pred = np.array([[2], [4], [4]], dtype=np.float32) >>> metric = keras.metrics.R2Score() >>> metric.update_state(y_true, y_pred) >>> result = metric.result() >>> result 0.57142854`

---
