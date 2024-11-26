---
title: CosineDecay
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/schedules/learning_rate_schedule.py#L572" >}}

### `CosineDecay` class

```python
keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps,
    alpha=0.0,
    name="CosineDecay",
    warmup_target=None,
    warmup_steps=0,
)
```

A `LearningRateSchedule` that uses a cosine decay with optional warmup.

See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
SGDR: Stochastic Gradient Descent with Warm Restarts.

For the idea of a linear warmup of our learning rate,
see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

When we begin training a model, we often want an initial increase in our
learning rate followed by a decay. If `warmup_target` is an int, this
schedule applies a linear increase per optimizer step to our learning rate
from `initial_learning_rate` to `warmup_target` for a duration of
`warmup_steps`. Afterwards, it applies a cosine decay function taking our
learning rate from `warmup_target` to `alpha` for a duration of
`decay_steps`. If `warmup_target` is None we skip warmup and our decay
will take our learning rate from `initial_learning_rate` to `alpha`.
It requires a `step` value to compute the learning rate. You can
just pass a backend variable that you increment at each training step.

The schedule is a 1-arg callable that produces a warmup followed by a
decayed learning rate when passed the current optimizer step. This can be
useful for changing the learning rate value across different invocations of
optimizer functions.

Our warmup is computed as:

```python
def warmup_learning_rate(step):
    completed_fraction = step / warmup_steps
    total_delta = target_warmup - initial_learning_rate
    return completed_fraction * total_delta
```

And our decay is computed as:

```python
if warmup_target is None:
    initial_decay_lr = initial_learning_rate
else:
    initial_decay_lr = warmup_target
def decayed_learning_rate(step):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_decay_lr * decayed
```

Example usage without warmup:

```python
decay_steps = 1000
initial_learning_rate = 0.1
lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps)
```

Example usage with warmup:

```python
decay_steps = 1000
initial_learning_rate = 0
warmup_steps = 1000
target_learning_rate = 0.1
lr_warmup_decayed_fn = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
    warmup_steps=warmup_steps
)
```

You can pass this schedule directly into a [`keras.optimizers.Optimizer`]({{< relref "/docs/api/optimizers#optimizer-class" >}})
as the learning rate. The learning rate schedule is also serializable and
deserializable using `keras.optimizers.schedules.serialize` and
`keras.optimizers.schedules.deserialize`.

**Arguments**

- **initial_learning_rate**: A Python float. The initial learning rate.
- **decay_steps**: A Python int. Number of steps to decay over.
- **alpha**: A Python float. Minimum learning rate value for decay as a
  fraction of `initial_learning_rate`.
- **name**: String. Optional name of the operation. Defaults to
  `"CosineDecay"`.
- **warmup_target**: A Python float. The target learning rate for our
  warmup phase. Will cast to the `initial_learning_rate` datatype.
  Setting to `None` will skip warmup and begins decay phase from
  `initial_learning_rate`. Otherwise scheduler will warmup from
  `initial_learning_rate` to `warmup_target`.
- **warmup_steps**: A Python int. Number of steps to warmup over.

**Returns**

A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the
same type as `initial_learning_rate`.
