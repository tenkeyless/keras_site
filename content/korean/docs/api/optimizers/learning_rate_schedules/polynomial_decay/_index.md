---
title: PolynomialDecay
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/schedules/learning_rate_schedule.py#L304" >}}

### `PolynomialDecay` class

```python
keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps,
    end_learning_rate=0.0001,
    power=1.0,
    cycle=False,
    name="PolynomialDecay",
)
```

A `LearningRateSchedule` that uses a polynomial decay schedule.

It is commonly observed that a monotonically decreasing learning rate, whose
degree of change is carefully chosen, results in a better performing model.
This schedule applies a polynomial decay function to an optimizer step,
given a provided `initial_learning_rate`, to reach an `end_learning_rate`
in the given `decay_steps`.

It requires a `step` value to compute the decayed learning rate. You
can just pass a backend variable that you increment at each training
step.

The schedule is a 1-arg callable that produces a decayed learning rate
when passed the current optimizer step. This can be useful for changing the
learning rate value across different invocations of optimizer functions.
It is computed as:

```python
def decayed_learning_rate(step):
    step = min(step, decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
```

If `cycle` is True then a multiple of `decay_steps` is used, the first one
that is bigger than `step`.

```python
def decayed_learning_rate(step):
    decay_steps = decay_steps * ceil(step / decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
```

You can pass this schedule directly into a [`keras.optimizers.Optimizer`]({{< relref "/docs/api/optimizers#optimizer-class" >}})
as the learning rate.
**Example**

Fit a model while decaying from 0.1 to 0.01 in 10000 steps using

sqrt (i.e. power=0.5):

```python
...
starter_learning_rate = 0.1
end_learning_rate = 0.01
decay_steps = 10000
learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)
model.compile(optimizer=keras.optimizers.SGD(
                  learning_rate=learning_rate_fn),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, epochs=5)
```

The learning rate schedule is also serializable and deserializable using
`keras.optimizers.schedules.serialize` and
`keras.optimizers.schedules.deserialize`.

**Arguments**

- **initial_learning_rate**: A Python float. The initial learning rate.
- **decay_steps**: A Python integer. Must be positive. See the decay
  computation above.
- **end_learning_rate**: A Python float. The minimal end learning rate.
- **power**: A Python float. The power of the polynomial. Defaults to
  `1.0`.
- **cycle**: A boolean, whether it should cycle beyond decay_steps.
- **name**: String. Optional name of the operation. Defaults to
  `"PolynomialDecay"`.

**Returns**

A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the
same type as `initial_learning_rate`.
