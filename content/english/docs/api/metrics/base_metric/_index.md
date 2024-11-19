---
title: base_metric
toc: false
---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/metric.py#L11)

### `Metric` class

`keras.metrics.Metric(dtype=None, name=None)`

Encapsulates metric logic and state.

**Arguments**

- **name**: Optional name for the metric instance.
- **dtype**: The dtype of the metric's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

**Example**

`m = SomeMetric(...) for input in ...:     m.update_state(input) print('Final result: ', m.result())`

Usage with `compile()` API:

`model = keras.Sequential() model.add(keras.layers.Dense(64, activation='relu')) model.add(keras.layers.Dense(64, activation='relu')) model.add(keras.layers.Dense(10, activation='softmax'))  model.compile(optimizer=keras.optimizers.RMSprop(0.01),               loss=keras.losses.CategoricalCrossentropy(),               metrics=[keras.metrics.CategoricalAccuracy()])  data = np.random.random((1000, 32)) labels = np.random.random((1000, 10))  model.fit(data, labels, epochs=10)`

To be implemented by subclasses:

- `__init__()`: All state variables should be created in this method by calling `self.add_variable()` like: `self.var = self.add_variable(...)`
- `update_state()`: Has all updates to the state variables like: `self.var.assign(...)`.
- `result()`: Computes and returns a scalar value or a dict of scalar values for the metric from the state variables.

Example subclass implementation:

`class BinaryTruePositives(Metric):      def __init__(self, name='binary_true_positives', **kwargs):         super().__init__(name=name, **kwargs)         self.true_positives = self.add_variable(             shape=(),             initializer='zeros',             name='true_positives'         )      def update_state(self, y_true, y_pred, sample_weight=None):         y_true = ops.cast(y_true, "bool")         y_pred = ops.cast(y_pred, "bool")          values = ops.logical_and(             ops.equal(y_true, True), ops.equal(y_pred, True))         values = ops.cast(values, self.dtype)         if sample_weight is not None:             sample_weight = ops.cast(sample_weight, self.dtype)             sample_weight = ops.broadcast_to(                 sample_weight, ops.shape(values)             )             values = ops.multiply(values, sample_weight)         self.true_positives.assign(self.true_positives + ops.sum(values))      def result(self):         return self.true_positives`

---
