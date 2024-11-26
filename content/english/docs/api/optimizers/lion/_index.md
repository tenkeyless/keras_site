---
title: Lion
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/lion.py#L6" >}}

### `Lion` class

```python
keras.optimizers.Lion(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.99,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="lion",
    **kwargs
)
```

Optimizer that implements the Lion algorithm.

The Lion optimizer is a stochastic-gradient-descent method that uses the
sign operator to control the magnitude of the update, unlike other adaptive
optimizers such as Adam that rely on second-order moments. This make
Lion more memory-efficient as it only keeps track of the momentum. According
to the authors (see reference), its performance gain over Adam grows with
the batch size. Because the update of Lion is produced through the sign
operation, resulting in a larger norm, a suitable learning rate for Lion is
typically 3-10x smaller than that for AdamW. The weight decay for Lion
should be in turn 3-10x larger than that for AdamW to maintain a
similar strength (lr \* wd).

**Arguments**

- **learning_rate**: A float, a
  [`keras.optimizers.schedules.LearningRateSchedule`]({{< relref "/docs/api/optimizers/learning_rate_schedules/learning_rate_schedule#learningrateschedule-class" >}}) instance, or
  a callable that takes no arguments and returns the actual value to
  use. The learning rate. Defaults to `0.001`.
- **beta_1**: A float value or a constant float tensor, or a callable
  that takes no arguments and returns the actual value to use. The
  rate to combine the current gradient and the 1st moment estimate.
  Defaults to `0.9`.
- **beta_2**: A float value or a constant float tensor, or a callable
  that takes no arguments and returns the actual value to use. The
  exponential decay rate for the 1st moment estimate. Defaults to
  `0.99`.
- **name**: String. The name to use
  for momentum accumulator weights created by
  the optimizer.
- **weight_decay**: Float. If set, weight decay is applied.
- **clipnorm**: Float. If set, the gradient of each weight is individually
  clipped so that its norm is no higher than this value.
- **clipvalue**: Float. If set, the gradient of each weight is clipped to be
  no higher than this value.
- **global_clipnorm**: Float. If set, the gradient of all weights is clipped
  so that their global norm is no higher than this value.
- **use_ema**: Boolean, defaults to `False`.
  If `True`, exponential moving average
  (EMA) is applied. EMA consists of computing an exponential moving
  average of the weights of the model (as the weight values change
  after each training batch), and periodically overwriting the
  weights with their moving average.
- **ema_momentum**: Float, defaults to 0.99. Only used if `use_ema=True`.
  This is the momentum to use when computing
  the EMA of the model's weights:
  `new_average = ema_momentum * old_average + (1 - ema_momentum) *
current_variable_value`.
- **ema_overwrite_frequency**: Int or None, defaults to None. Only used if
  `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
  we overwrite the model variable by its moving average.
  If None, the optimizer
  does not overwrite model variables in the middle of training,
  and you need to explicitly overwrite the variables
  at the end of training by calling
  `optimizer.finalize_variable_values()` (which updates the model
  variables in-place). When using the built-in `fit()` training loop,
  this happens automatically after the last epoch,
  and you don't need to do anything.
- **loss_scale_factor**: Float or `None`. If a float, the scale factor will
  be multiplied the loss before computing gradients, and the inverse
  of the scale factor will be multiplied by the gradients before
  updating variables. Useful for preventing underflow during
  mixed precision training. Alternately,
  [`keras.optimizers.LossScaleOptimizer`]({{< relref "/docs/api/optimizers/loss_scale_optimizer#lossscaleoptimizer-class" >}}) will
  automatically set a loss scale factor.
- **gradient_accumulation_steps**: Int or `None`. If an int, model & optimizer
  variables will not be updated at every step; instead they will be
  updated every `gradient_accumulation_steps` steps, using the average
  value of the gradients since the last update. This is known as
  "gradient accumulation". This can be useful
  when your batch size is very small, in order to reduce gradient
  noise at each update step. EMA frequency will look at "accumulated"
  iterations value (optimizer steps // gradient_accumulation_steps).
  Learning rate schedules will look at "real" iterations value
  (optimizer steps).

**References**

- [Chen et al., 2023](http://arxiv.org/abs/2302.06675)
- [Authors' implementation](http://github.com/google/automl/tree/master/lion)
