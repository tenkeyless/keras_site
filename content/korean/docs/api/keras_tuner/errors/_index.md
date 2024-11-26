---
title: Errors
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/errors.py#L19" >}}

### `FailedTrialError` class

```python
keras_tuner.errors.FailedTrialError()
```

Raise this error to mark a `Trial` as failed.

When this error is raised in a `Trial`, the `Tuner` would not retry the
`Trial` but directly mark it as `"FAILED"`.

**Example**

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        # Build the model
        ...
        if too_slow(model):
            # Mark the Trial as "FAILED" if the model is too slow.
            raise keras_tuner.FailedTrialError("Model is too slow.")
        return model
```

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/errors.py#L43" >}}

### `FatalError` class

```python
keras_tuner.errors.FatalError()
```

A fatal error during search to terminate the program.

It is used to terminate the KerasTuner program for errors that need
users immediate attention. When this error is raised in a `Trial`, it will
not be caught by KerasTuner.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/errors.py#L55" >}}

### `FatalValueError` class

```python
keras_tuner.errors.FatalValueError()
```

A fatal error during search to terminate the program.

It is a subclass of `FatalError` and `ValueError`.

It is used to terminate the KerasTuner program for errors that need
users immediate attention. When this error is raised in a `Trial`, it will
not be caught by KerasTuner.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/errors.py#L69" >}}

### `FatalTypeError` class

```python
keras_tuner.errors.FatalTypeError()
```

A fatal error during search to terminate the program.

It is a subclass of `FatalError` and `TypeError`.

It is used to terminate the KerasTuner program for errors that need
users immediate attention. When this error is raised in a `Trial`, it will
not be caught by KerasTuner.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/errors.py#L83" >}}

### `FatalRuntimeError` class

```python
keras_tuner.errors.FatalRuntimeError()
```

A fatal error during search to terminate the program.

It is a subclass of `FatalError` and `RuntimeError`.

It is used to terminate the KerasTuner program for errors that need
users immediate attention. When this error is raised in a `Trial`, it will
not be caught by KerasTuner.
