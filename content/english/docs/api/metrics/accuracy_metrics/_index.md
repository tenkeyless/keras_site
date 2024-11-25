---
title: Accuracy metrics
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/accuracy_metrics.py#L15" >}}

### `Accuracy` class

```python
keras.metrics.Accuracy(name="accuracy", dtype=None)
```

Calculates how often predictions equal labels.

This metric creates two local variables, `total` and `count` that are used
to compute the frequency with which `y_pred` matches `y_true`. This
frequency is ultimately returned as `binary accuracy`: an idempotent
operation that simply divides `total` by `count`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Examples**

```console
>>> m = keras.metrics.Accuracy()
>>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
>>> m.result()
0.75
```

```console
>>> m.reset_state()
>>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
...                sample_weight=[1, 1, 0, 0])
>>> m.result()
0.5
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=[keras.metrics.Accuracy()])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/accuracy_metrics.py#L72" >}}

### `BinaryAccuracy` class

```python
keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
```

Calculates how often predictions match binary labels.

This metric creates two local variables, `total` and `count` that are used
to compute the frequency with which `y_pred` matches `y_true`. This
frequency is ultimately returned as `binary accuracy`: an idempotent
operation that simply divides `total` by `count`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.
- **threshold**: (Optional) Float representing the threshold for deciding
  whether prediction values are 1 or 0.

**Example**

```console
>>> m = keras.metrics.BinaryAccuracy()
>>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
>>> m.result()
0.75
```

```console
>>> m.reset_state()
>>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]],
...                sample_weight=[1, 0, 0, 1])
>>> m.result()
0.5
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=[keras.metrics.BinaryAccuracy()])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/accuracy_metrics.py#L160" >}}

### `CategoricalAccuracy` class

```python
keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)
```

Calculates how often predictions match one-hot labels.

You can provide logits of classes as `y_pred`, since argmax of
logits and probabilities are same.

This metric creates two local variables, `total` and `count` that are used
to compute the frequency with which `y_pred` matches `y_true`. This
frequency is ultimately returned as `categorical accuracy`: an idempotent
operation that simply divides `total` by `count`.

`y_pred` and `y_true` should be passed in as vectors of probabilities,
rather than as labels. If necessary, use `ops.one_hot` to expand `y_true` as
a vector.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.CategoricalAccuracy()
>>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
...                 [0.05, 0.95, 0]])
>>> m.result()
0.5
```

```console
>>> m.reset_state()
>>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
...                 [0.05, 0.95, 0]],
...                sample_weight=[0.7, 0.3])
>>> m.result()
0.3
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=[keras.metrics.CategoricalAccuracy()])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/accuracy_metrics.py#L249" >}}

### `SparseCategoricalAccuracy` class

```python
keras.metrics.SparseCategoricalAccuracy(
    name="sparse_categorical_accuracy", dtype=None
)
```

Calculates how often predictions match integer labels.

```python
acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
```

You can provide logits of classes as `y_pred`, since argmax of
logits and probabilities are same.

This metric creates two local variables, `total` and `count` that are used
to compute the frequency with which `y_pred` matches `y_true`. This
frequency is ultimately returned as `sparse categorical accuracy`: an
idempotent operation that simply divides `total` by `count`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.SparseCategoricalAccuracy()
>>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
>>> m.result()
0.5
```

```console
>>> m.reset_state()
>>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]],
...                sample_weight=[0.7, 0.3])
>>> m.result()
0.3
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/accuracy_metrics.py#L333" >}}

### `TopKCategoricalAccuracy` class

```python
keras.metrics.TopKCategoricalAccuracy(
    k=5, name="top_k_categorical_accuracy", dtype=None
)
```

Computes how often targets are in the top `K` predictions.

**Arguments**

- **k**: (Optional) Number of top elements to look at for computing accuracy.
  Defaults to `5`.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.TopKCategoricalAccuracy(k=1)
>>> m.update_state([[0, 0, 1], [0, 1, 0]],
...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
>>> m.result()
0.5
```

```console
>>> m.reset_state()
>>> m.update_state([[0, 0, 1], [0, 1, 0]],
...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
...                sample_weight=[0.7, 0.3])
>>> m.result()
0.3
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=[keras.metrics.TopKCategoricalAccuracy()])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/accuracy_metrics.py#L411" >}}

### `SparseTopKCategoricalAccuracy` class

```python
keras.metrics.SparseTopKCategoricalAccuracy(
    k=5, name="sparse_top_k_categorical_accuracy", dtype=None
)
```

Computes how often integer targets are in the top `K` predictions.

**Arguments**

- **k**: (Optional) Number of top elements to look at for computing accuracy.
  Defaults to `5`.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.SparseTopKCategoricalAccuracy(k=1)
>>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
>>> m.result()
0.5
```

```console
>>> m.reset_state()
>>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
...                sample_weight=[0.7, 0.3])
>>> m.result()
0.3
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=[keras.metrics.SparseTopKCategoricalAccuracy()])
```
