---
title: CSVLogger
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/callbacks/csv_logger.py#L11" >}}

### `CSVLogger` class

```python
keras.callbacks.CSVLogger(filename, separator=",", append=False)
```

Callback that streams epoch results to a CSV file.

Supports all values that can be represented as a string,
including 1D iterables such as `np.ndarray`.

**Arguments**

- **filename**: Filename of the CSV file, e.g. `'run/log.csv'`.
- **separator**: String used to separate elements in the CSV file.
- **append**: Boolean. True: append if file exists (useful for continuing
  training). False: overwrite existing file.

**Example**

```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```
