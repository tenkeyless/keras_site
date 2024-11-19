---
title: base_layer
toc: false
---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/layer.py#L61)

### `Layer` class

`keras.layers.Layer(     activity_regularizer=None,     trainable=True,     dtype=None,     autocast=True,     name=None,     **kwargs )`

This is the class from which all layers inherit.

A layer is a callable object that takes as input one or more tensors and that outputs one or more tensors. It involves _computation_, defined in the `call()` method, and a _state_ (weight variables). State can be created:

- in `__init__()`, for instance via `self.add_weight()`;
- in the optional `build()` method, which is invoked by the first `__call__()` to the layer, and supplies the shape(s) of the input(s), which may not have been known at initialization time.

Layers are recursively composable: If you assign a Layer instance as an attribute of another Layer, the outer layer will start tracking the weights created by the inner layer. Nested layers should be instantiated in the `__init__()` method or `build()` method.

Users will just instantiate a layer and then treat it as a callable.

**Arguments**

- **trainable**: Boolean, whether the layer's variables should be trainable.
- **name**: String name of the layer.
- **dtype**: The dtype of the layer's computations and weights. Can also be a `keras.DTypePolicy`, which allows the computation and weight dtype to differ. Defaults to `None`. `None` means to use `keras.config.dtype_policy()`, which is a `float32` policy unless set to different value (via `keras.config.set_dtype_policy()`).

**Attributes**

- **name**: The name of the layer (string).
- **dtype**: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
- **variable_dtype**: Dtype of the layer's weights.
- **compute_dtype**: The dtype of the layer's computations. Layers automatically cast inputs to this dtype, which causes the computations and output to also be in this dtype. When mixed precision is used with a `keras.DTypePolicy`, this will be different than `variable_dtype`.
- **trainable_weights**: List of variables to be included in backprop.
- **non_trainable_weights**: List of variables that should not be included in backprop.
- **weights**: The concatenation of the lists trainable_weights and non_trainable_weights (in this order).
- **trainable**: Whether the layer should be trained (boolean), i.e. whether its potentially-trainable weights should be returned as part of `layer.trainable_weights`.
- **input_spec**: Optional (list of) `InputSpec` object(s) specifying the constraints on inputs that can be accepted by the layer.

We recommend that descendants of `Layer` implement the following methods:

- `__init__()`: Defines custom layer attributes, and creates layer weights that do not depend on input shapes, using `add_weight()`, or other state.
- `build(self, input_shape)`: This method can be used to create weights that depend on the shape(s) of the input(s), using `add_weight()`, or other state. `__call__()` will automatically build the layer (if it has not been built yet) by calling `build()`.
- `call(self, *args, **kwargs)`: Called in `__call__` after making sure `build()` has been called. `call()` performs the logic of applying the layer to the input arguments. Two reserved keyword arguments you can optionally use in `call()` are: 1. `training` (boolean, whether the call is in inference mode or training mode). 2. `mask` (boolean tensor encoding masked timesteps in the input, used e.g. in RNN layers). A typical signature for this method is `call(self, inputs)`, and user could optionally add `training` and `mask` if the layer need them.
- `get_config(self)`: Returns a dictionary containing the configuration used to initialize this layer. If the keys differ from the arguments in `__init__()`, then override `from_config(self)` as well. This method is used when saving the layer or a model that contains this layer.

**Examples**

Here's a basic example: a layer with two variables, `w` and `b`, that returns `y = w . x + b`. It shows how to implement `build()` and `call()`. Variables set as attributes of a layer are tracked as weights of the layers (in `layer.weights`).

`` class SimpleDense(Layer):     def __init__(self, units=32):         super().__init__()         self.units = units      # Create the state of the layer (weights)     def build(self, input_shape):         self.kernel = self.add_weight(             shape=(input_shape[-1], self.units),             initializer="glorot_uniform",             trainable=True,             name="kernel",         )         self.bias = self.add_weight(             shape=(self.units,),             initializer="zeros",             trainable=True,             name="bias",         )      # Defines the computation     def call(self, inputs):         return ops.matmul(inputs, self.kernel) + self.bias  # Instantiates the layer. linear_layer = SimpleDense(4)  # This will also call `build(input_shape)` and create the weights. y = linear_layer(ops.ones((2, 2))) assert len(linear_layer.weights) == 2  # These weights are trainable, so they're listed in `trainable_weights`: assert len(linear_layer.trainable_weights) == 2 ``

Besides trainable weights, updated via backpropagation during training, layers can also have non-trainable weights. These weights are meant to be updated manually during `call()`. Here's a example layer that computes the running sum of its inputs:

`class ComputeSum(Layer):    def __init__(self, input_dim):       super(ComputeSum, self).__init__()       # Create a non-trainable weight.       self.total = self.add_weight(         shape=(),         initializer="zeros",         trainable=False,         name="total",       )    def call(self, inputs):       self.total.assign(self.total + ops.sum(inputs))       return self.total  my_sum = ComputeSum(2) x = ops.ones((2, 2)) y = my_sum(x)  assert my_sum.weights == [my_sum.total] assert my_sum.non_trainable_weights == [my_sum.total] assert my_sum.trainable_weights == []`

---

### `weights` property

`keras.layers.Layer.weights`

List of all weight variables of the layer.

Unlike, `layer.variables` this excludes metric state and random seeds.

---

### `trainable_weights` property

`keras.layers.Layer.trainable_weights`

List of all trainable weight variables of the layer.

These are the weights that get updated by the optimizer during training.

---

### `non_trainable_weights` property

`keras.layers.Layer.non_trainable_weights`

List of all non-trainable weight variables of the layer.

These are the weights that should not be updated by the optimizer during training. Unlike, `layer.non_trainable_variables` this excludes metric state and random seeds.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/layer.py#L485)

### `add_weight` method

`Layer.add_weight(     shape=None,     initializer=None,     dtype=None,     trainable=True,     autocast=True,     regularizer=None,     constraint=None,     aggregation="mean",     name=None, )`

Add a weight variable to the layer.

**Arguments**

- **shape**: Shape tuple for the variable. Must be fully-defined (no `None` entries). Defaults to `()` (scalar) if unspecified.
- **initializer**: Initializer object to use to populate the initial variable value, or string name of a built-in initializer (e.g. `"random_normal"`). If unspecified, defaults to `"glorot_uniform"` for floating-point variables and to `"zeros"` for all other types (e.g. int, bool).
- **dtype**: Dtype of the variable to create, e.g. `"float32"`. If unspecified, defaults to the layer's variable dtype (which itself defaults to `"float32"` if unspecified).
- **trainable**: Boolean, whether the variable should be trainable via backprop or whether its updates are managed manually. Defaults to `True`.
- **autocast**: Boolean, whether to autocast layers variables when accessing them. Defaults to `True`.
- **regularizer**: Regularizer object to call to apply penalty on the weight. These penalties are summed into the loss function during optimization. Defaults to `None`.
- **constraint**: Contrainst object to call on the variable after any optimizer update, or string name of a built-in constraint. Defaults to `None`.
- **aggregation**: String, one of `'mean'`, `'sum'`, `'only_first_replica'`. Annotates the variable with the type of multi-replica aggregation to be used for this variable when writing custom data parallel training loops.
- **name**: String name of the variable. Useful for debugging purposes.

---

### `trainable` property

`keras.layers.Layer.trainable`

Settable boolean, whether this layer should be trainable or not.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/layer.py#L688)

### `get_weights` method

`Layer.get_weights()`

Return the values of `layer.weights` as a list of NumPy arrays.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/layer.py#L692)

### `set_weights` method

`Layer.set_weights(weights)`

Sets the values of `layer.weights` from a list of NumPy arrays.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/layer.py#L1531)

### `get_config` method

`Model.get_config()`

Returns the config of the object.

An object config is a Python dictionary (serializable) containing the information needed to re-instantiate it.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/layer.py#L1104)

### `add_loss` method

`Layer.add_loss(loss)`

Can be called inside of the `call()` method to add a scalar loss.

**Example**

`class MyLayer(Layer):     ...     def call(self, x):         self.add_loss(ops.sum(x))         return x`

---

### `losses` property

`keras.layers.Layer.losses`

List of scalar losses from `add_loss`, regularizers and sublayers.

---
