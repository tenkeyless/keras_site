---
title: Optimizers
toc: true
weight: 5
type: docs
---

## Available optimizers

- [SGD](/api/optimizers/sgd/)
- [RMSprop](/api/optimizers/rmsprop/)
- [Adam](/api/optimizers/adam/)
- [AdamW](/api/optimizers/adamw/)
- [Adadelta](/api/optimizers/adadelta/)
- [Adagrad](/api/optimizers/adagrad/)
- [Adamax](/api/optimizers/adamax/)
- [Adafactor](/api/optimizers/adafactor/)
- [Nadam](/api/optimizers/Nadam/)
- [Ftrl](/api/optimizers/ftrl/)
- [Lion](/api/optimizers/lion/)
- [Lamb](/api/optimizers/lamb/)
- [Loss Scale Optimizer](/api/optimizers/loss_scale_optimizer/)

---

## Usage with `compile()` & `fit()`

An optimizer is one of the two arguments required for compiling a Keras model:

`import keras from keras import layers  model = keras.Sequential() model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,))) model.add(layers.Activation('softmax'))  opt = keras.optimizers.Adam(learning_rate=0.01) model.compile(loss='categorical_crossentropy', optimizer=opt)`

You can either instantiate an optimizer before passing it to `model.compile()` , as in the above example, or you can pass it by its string identifier. In the latter case, the default parameters for the optimizer will be used.

`# pass optimizer by name: default parameters will be used model.compile(loss='categorical_crossentropy', optimizer='adam')`

---

## Learning rate decay / scheduling

You can use a [learning rate schedule](/api/optimizers/learning_rate_schedules) to modulate how the learning rate of your optimizer changes over time:

`lr_schedule = keras.optimizers.schedules.ExponentialDecay(     initial_learning_rate=1e-2,     decay_steps=10000,     decay_rate=0.9) optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)`

Check out [the learning rate schedule API documentation](/api/optimizers/learning_rate_schedules) for a list of available schedules.

---

## Base Optimizer API

These methods and attributes are common to all Keras optimizers.

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/optimizer.py#L21)

### `Optimizer` class

`keras.optimizers.Optimizer()`

Abstract optimizer base class.

If you intend to create your own optimization algorithm, please inherit from this class and override the following methods:

- `build`: Create your optimizer-related variables, such as momentum variables in the SGD optimizer.
- `update_step`: Implement your optimizer's variable updating logic.
- `get_config`: serialization of the optimizer.

**Example**

`class SGD(Optimizer):     def __init__(self, **kwargs):         super().__init__(**kwargs)         self.momentum = 0.9      def build(self, variables):         super().build(variables)         self.momentums = []         for variable in variables:             self.momentums.append(                 self.add_variable_from_reference(                     reference_variable=variable, name="momentum"                 )             )      def update_step(self, gradient, variable, learning_rate):         learning_rate = ops.cast(learning_rate, variable.dtype)         gradient = ops.cast(gradient, variable.dtype)         m = self.momentums[self._get_variable_index(variable)]         self.assign(             m,             ops.subtract(                 ops.multiply(m, ops.cast(self.momentum, variable.dtype)),                 ops.multiply(gradient, learning_rate),             ),         )         self.assign_add(variable, m)      def get_config(self):         config = super().get_config()         config.update(             {                 "momentum": self.momentum,                 "nesterov": self.nesterov,             }         )         return config`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/optimizers/base_optimizer.py#L342)

### `apply_gradients` method

`Optimizer.apply_gradients(grads_and_vars)`

---

### `variables` property

`keras.optimizers.Optimizer.variables`

---
