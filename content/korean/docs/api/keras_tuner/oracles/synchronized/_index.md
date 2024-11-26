---
title: "@synchronized decorator"
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/oracle.py#L44" >}}

### `synchronized` function

```python
keras_tuner.synchronized(func, )
```

Decorator to synchronize the multi-threaded calls to `Oracle` functions.

In parallel tuning, there may be concurrent gRPC calls from multiple threads
to the `Oracle` methods like `create_trial()`, `update_trial()`, and
`end_trial()`. To avoid concurrent writing to the data, use `@synchronized`
to ensure the calls are synchronized, which only allows one call to run at a
time.

Concurrent calls to different `Oracle` objects would not block one another.
Concurrent calls to the same or different functions of the same `Oracle`
object would block one another.

You can decorate a subclass function, which overrides an already decorated
function in the base class, without worrying about creating a deadlock.
However, the decorator only support methods within classes, and cannot be
applied to standalone functions.

You do not need to decorate `Oracle.populate_space()`, which is only
called by `Oracle.create_trial()`, which is decorated.

**Example**

```python
class MyOracle(keras_tuner.Oracle):
    @keras_tuner.synchronized
    def create_trial(self, tuner_id):
        super().create_trial(tuner_id)
        ...
    @keras_tuner.synchronized
    def update_trial(self, trial_id, metrics, step=0):
        super().update_trial(trial_id, metrics, step)
        ...
    @keras_tuner.synchronized
    def end_trial(self, trial):
        super().end_trial(trial)
        ...
```
