---
title: Adam
toc: true
weight: 3
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/adam.py#L6" >}}

### `Adam` class

`keras.optimizers.Adam(     learning_rate=0.001,     beta_1=0.9,     beta_2=0.999,     epsilon=1e-07,     amsgrad=False,     weight_decay=None,     clipnorm=None,     clipvalue=None,     global_clipnorm=None,     use_ema=False,     ema_momentum=0.99,     ema_overwrite_frequency=None,     loss_scale_factor=None,     gradient_accumulation_steps=None,     name="adam",     **kwargs )`

Optimizer that implements the Adam algorithm.

Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.

According to [Kingma et al., 2014](http://arxiv.org/abs/1412.6980), the method is "_computationally efficient, has little memory requirement, invariant to diagonal rescaling of gradients, and is well suited for problems that are large in terms of data/parameters_".

**Arguments**

- **learning_rate**: A float, a [`keras.optimizers.schedules.LearningRateSchedule`](/api/optimizers/learning_rate_schedules/learning_rate_schedule#learningrateschedule-class) instance, or a callable that takes no arguments and returns the actual value to use. The learning rate. Defaults to `0.001`.
- **beta_1**: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual value to use. The exponential decay rate for the 1st moment estimates. Defaults to `0.9`.
- **beta_2**: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual value to use. The exponential decay rate for the 2nd moment estimates. Defaults to `0.999`.
- **epsilon**: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to `1e-7`.
- **amsgrad**: Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond". Defaults to `False`.
- **name**: String. The name to use for momentum accumulator weights created by the optimizer.
- **weight_decay**: Float. If set, weight decay is applied.
- **clipnorm**: Float. If set, the gradient of each weight is individually clipped so that its norm is no higher than this value.
- **clipvalue**: Float. If set, the gradient of each weight is clipped to be no higher than this value.
- **global_clipnorm**: Float. If set, the gradient of all weights is clipped so that their global norm is no higher than this value.
- **use_ema**: Boolean, defaults to `False`. If `True`, exponential moving average (EMA) is applied. EMA consists of computing an exponential moving average of the weights of the model (as the weight values change after each training batch), and periodically overwriting the weights with their moving average.
- **ema_momentum**: Float, defaults to 0.99. Only used if `use_ema=True`. This is the momentum to use when computing the EMA of the model's weights: `new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value`.
- **ema_overwrite_frequency**: Int or None, defaults to None. Only used if `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations, we overwrite the model variable by its moving average. If None, the optimizer does not overwrite model variables in the middle of training, and you need to explicitly overwrite the variables at the end of training by calling `optimizer.finalize_variable_values()` (which updates the model variables in-place). When using the built-in `fit()` training loop, this happens automatically after the last epoch, and you don't need to do anything.
- **loss_scale_factor**: Float or `None`. If a float, the scale factor will be multiplied the loss before computing gradients, and the inverse of the scale factor will be multiplied by the gradients before updating variables. Useful for preventing underflow during mixed precision training. Alternately, [`keras.optimizers.LossScaleOptimizer`](/api/optimizers/loss_scale_optimizer#lossscaleoptimizer-class) will automatically set a loss scale factor.
- **gradient_accumulation_steps**: Int or `None`. If an int, model & optimizer variables will not be updated at every step; instead they will be updated every `gradient_accumulation_steps` steps, using the average value of the gradients since the last update. This is known as "gradient accumulation". This can be useful when your batch size is very small, in order to reduce gradient noise at each update step. EMA frequency will look at "accumulated" iterations value (optimizer steps // gradient_accumulation_steps). Learning rate schedules will look at "real" iterations value (optimizer steps).

---
