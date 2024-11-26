---
title: LearningRateSchedule
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/schedules/learning_rate_schedule.py#L10" >}}

### `LearningRateSchedule` class

```python
keras.optimizers.schedules.LearningRateSchedule()
```

The learning rate schedule base class.

You can use a learning rate schedule to modulate how the learning rate
of your optimizer changes over time.

Several built-in learning rate schedules are available, such as
[`keras.optimizers.schedules.ExponentialDecay`]({{< relref "/docs/api/optimizers/learning_rate_schedules/exponential_decay#exponentialdecay-class" >}}) or
[`keras.optimizers.schedules.PiecewiseConstantDecay`]({{< relref "/docs/api/optimizers/learning_rate_schedules/piecewise_constant_decay#piecewiseconstantdecay-class" >}}):

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
```

A `LearningRateSchedule` instance can be passed in as the `learning_rate`
argument of any optimizer.

To implement your own schedule object, you should implement the `__call__`
method, which takes a `step` argument (scalar integer tensor, the
current training step count).
Like for any other Keras object, you can also optionally
make your object serializable by implementing the `get_config`
and `from_config` methods.

**Example**

```python
class MyLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
    def __call__(self, step):
        return self.initial_learning_rate / (step + 1)
optimizer = keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))
```
