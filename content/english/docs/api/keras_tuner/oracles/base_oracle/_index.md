---
title: The base Oracle class
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L251" >}}

### `Oracle` class

```python
keras_tuner.Oracle(
    objective=None,
    max_trials=None,
    hyperparameters=None,
    allow_new_entries=True,
    tune_new_entries=True,
    seed=None,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
)
```

Implements a hyperparameter optimization algorithm.

In a parallel tuning setting, there is only one `Oracle` instance. The
workers would communicate with the centralized `Oracle` instance with gPRC
calls to the `Oracle` methods.

`Trial` objects are often used as the communication packet through the gPRC
calls to pass information between the worker `Tuner` instances and the
`Oracle`. For example, `Oracle.create_trial()` returns a `Trial` object, and
`Oracle.end_trial()` accepts a `Trial` in its arguments.

New copies of the same `Trial` instance are reconstructed as it going
through the gRPC calls. The changes to the `Trial` objects in the worker
`Tuner`s are synced to the original copy in the `Oracle` when they are
passed back to the `Oracle` by calling `Oracle.end_trial()`.

**Arguments**

- **objective**: A string, [`keras_tuner.Objective`]({{< relref "/docs/api/keras_tuner/tuners/objective#objective-class" >}}) instance, or a list of
  [`keras_tuner.Objective`]({{< relref "/docs/api/keras_tuner/tuners/objective#objective-class" >}})s and strings. If a string, the direction of
  the optimization (min or max) will be inferred. If a list of
  [`keras_tuner.Objective`]({{< relref "/docs/api/keras_tuner/tuners/objective#objective-class" >}}), we will minimize the sum of all the
  objectives to minimize subtracting the sum of all the objectives to
  maximize. The `objective` argument is optional when
  `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
  the objective to minimize.
- **max_trials**: Integer, the total number of trials (model configurations)
  to test at most. Note that the oracle may interrupt the search
  before `max_trial` models have been tested if the search space has
  been exhausted.
- **hyperparameters**: Optional `HyperParameters` instance. Can be used to
  override (or register in advance) hyperparameters in the search
  space.
- **tune_new_entries**: Boolean, whether hyperparameter entries that are
  requested by the hypermodel but that were not specified in
  `hyperparameters` should be added to the search space, or not. If
  not, then the default value for these parameters will be used.
  Defaults to True.
- **allow_new_entries**: Boolean, whether the hypermodel is allowed to
  request hyperparameter entries not listed in `hyperparameters`.
  Defaults to True.
- **seed**: Int. Random seed.
- **max_retries_per_trial**: Integer. Defaults to 0. The maximum number of
  times to retry a `Trial` if the trial crashed or the results are
  invalid.
- **max_consecutive_failed_trials**: Integer. Defaults to 3. The maximum
  number of consecutive failed `Trial`s. When this number is reached,
  the search will be stopped. A `Trial` is marked as failed when none
  of the retries succeeded.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L91" >}}

### `wrapped_func` function

```python
keras_tuner.Oracle.create_trial()
```

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L91" >}}

### `wrapped_func` function

```python
keras_tuner.Oracle.end_trial()
```

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L663" >}}

### `get_best_trials` method

```python
Oracle.get_best_trials(num_trials=1)
```

Returns the best `Trial`s.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L692" >}}

### `get_state` method

```python
Oracle.get_state()
```

Returns the current state of this object.

This method is called during `save`.

**Returns**

A dictionary of serializable objects as the state.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L714" >}}

### `set_state` method

```python
Oracle.set_state(state)
```

Sets the current state of this object.

This method is called during `reload`.

**Arguments**

- **state**: A dictionary of serialized objects as the state to restore.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L423" >}}

### `score_trial` method

```python
Oracle.score_trial(trial)
```

Score a completed `Trial`.

This method can be overridden in subclasses to provide a score for
a set of hyperparameter values. This method is called from `end_trial`
on completed `Trial`s.

**Arguments**

- **trial**: A completed `Trial` object.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L395" >}}

### `populate_space` method

```python
Oracle.populate_space(trial_id)
```

Fill the hyperparameter space with values for a trial.

This method should be overridden in subclasses and called in
`create_trial` in order to populate the hyperparameter space with
values.

**Arguments**

- **trial_id**: A string, the ID for this Trial.

**Returns**

A dictionary with keys "values" and "status", where "values" is
a mapping of parameter names to suggested values, and "status"
should be one of "RUNNING" (the trial can start normally), "IDLE"
(the oracle is waiting on something and cannot create a trial), or
"STOPPED" (the oracle has finished searching and no new trial should
be created).

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L91" >}}

### `wrapped_func` function

```python
keras_tuner.Oracle.update_trial()
```
