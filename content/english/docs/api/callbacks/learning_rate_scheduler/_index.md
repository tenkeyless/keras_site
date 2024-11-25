---
title: LearningRateScheduler
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/callbacks/learning_rate_scheduler.py#L9" >}}

### `LearningRateScheduler` class

```python
keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

Learning rate scheduler.

At the beginning of every epoch, this callback gets the updated learning
rate value from `schedule` function provided at `__init__`, with the current
epoch and current learning rate, and applies the updated learning rate on
the optimizer.

**Arguments**

- **schedule**: A function that takes an epoch index (integer, indexed from 0)
  and current learning rate (float) as inputs and returns a new
  learning rate as output (float).
- **verbose**: Integer. 0: quiet, 1: log update messages.

**Example**

```console
>>> # This function keeps the initial learning rate for the first ten epochs
>>> # and decreases it exponentially after that.
>>> def scheduler(epoch, lr):
...     if epoch < 10:
...         return lr
...     else:
...         return lr * ops.exp(-0.1)
>>>
>>> model = keras.models.Sequential([keras.layers.Dense(10)])
>>> model.compile(keras.optimizers.SGD(), loss='mse')
>>> round(model.optimizer.learning_rate, 5)
0.01
```

```console
>>> callback = keras.callbacks.LearningRateScheduler(scheduler)
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
...                     epochs=15, callbacks=[callback], verbose=0)
>>> round(model.optimizer.learning_rate, 5)
0.00607
```
