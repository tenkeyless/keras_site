---
title: Adamax
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/adamax.py#L6" >}}

### `Adamax` class

```python
keras.optimizers.Adamax(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="adamax",
    **kwargs
)
```

Optimizer that implements the Adamax algorithm.

Adamax, a variant of Adam based on the infinity norm, is a first-order
gradient-based optimization method. Due to its capability of adjusting the
learning rate based on data characteristics, it is suited to learn
time-variant process, e.g., speech data with dynamically changed noise
conditions. Default parameters follow those provided in the paper (see
references below).

Initialization:

```python
m = 0  # Initialize initial 1st moment vector
u = 0  # Initialize the exponentially weighted infinity norm
t = 0  # Initialize timestep
```

The update rule for parameter `w` with gradient `g` is described at the end
of section 7.1 of the paper (see the reference section):

```python
t += 1
m = beta1 * m + (1 - beta) * g
u = max(beta2 * u, abs(g))
current_lr = learning_rate / (1 - beta1 ** t)
w = w - current_lr * m / (u + epsilon)
```

**Arguments**

- **learning_rate**: A float, a
  [`keras.optimizers.schedules.LearningRateSchedule`]({{< relref "/docs/api/optimizers/learning_rate_schedules/learning_rate_schedule#learningrateschedule-class" >}}) instance, or
  a callable that takes no arguments and returns the actual value to
  use. The learning rate. Defaults to `0.001`.
- **beta_1**: A float value or a constant float tensor. The exponential decay
  rate for the 1st moment estimates.
- **beta_2**: A float value or a constant float tensor. The exponential decay
  rate for the exponentially weighted infinity norm.
- **epsilon**: A small constant for numerical stability.
  name: String. The name to use
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

**Reference**

- [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
