---
title: Layer weight regularizers
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

Regularizers allow you to apply penalties on layer parameters or layer activity during optimization. These penalties are summed into the loss function that the network optimizes.

Regularization penalties are applied on a per-layer basis. The exact API will depend on the layer, but many layers (e.g. `Dense`, `Conv1D`, `Conv2D` and `Conv3D`) have a unified API.

These layers expose 3 keyword arguments:

- `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
- `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
- `activity_regularizer`: Regularizer to apply a penalty on the layer's output

```python
from keras import layers
from keras import regularizers

layer = layers.Dense(
    units=64,
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)
)
```

The value returned by the `activity_regularizer` object gets divided by the input batch size so that the relative weighting between the weight regularizers and the activity regularizers does not change with the batch size.

You can access a layer's regularization penalties by calling `layer.losses` after calling the layer on inputs:

```python
from keras import ops

layer = layers.Dense(units=5,
                     kernel_initializer='ones',
                     kernel_regularizer=regularizers.L1(0.01),
                     activity_regularizer=regularizers.L2(0.01))
tensor = ops.ones(shape=(5, 5)) * 2.0
out = layer(tensor)
# The kernel regularization term is 0.25
# The activity regularization term (after dividing by the batch size) is 5
print(ops.sum(layer.losses))  # 5.25 (= 5 + 0.25)
```

## Available regularizers

The following built-in regularizers are available as part of the `keras.regularizers` module:

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/regularizers/regularizers.py#L8" >}}

### `Regularizer` class

```python
keras.regularizers.Regularizer()
```

Regularizer base class.

Regularizers allow you to apply penalties on layer parameters or layer activity during optimization. These penalties are summed into the loss function that the network optimizes.

Regularization penalties are applied on a per-layer basis. The exact API will depend on the layer, but many layers (e.g. `Dense`, `Conv1D`, `Conv2D` and `Conv3D`) have a unified API.

These layers expose 3 keyword arguments:

- `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
- `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
- `activity_regularizer`: Regularizer to apply a penalty on the layer's output

All layers (including custom layers) expose `activity_regularizer` as a settable property, whether or not it is in the constructor arguments.

The value returned by the `activity_regularizer` is divided by the input batch size so that the relative weighting between the weight regularizers and the activity regularizers does not change with the batch size.

You can access a layer's regularization penalties by calling `layer.losses` after calling the layer on inputs.

## Example

```console
>>> layer = Dense(
...     5, input_dim=5,
...     kernel_initializer='ones',
...     kernel_regularizer=L1(0.01),
...     activity_regularizer=L2(0.01))
>>> tensor = ops.ones(shape=(5, 5)) * 2.0
>>> out = layer(tensor)
```

```console
>>> # The kernel regularization term is 0.25
>>> # The activity regularization term (after dividing by the batch size)
>>> # is 5
>>> ops.sum(layer.losses)
5.25
```

## Available penalties

```python
L1(0.3)  # L1 Regularization Penalty
L2(0.1)  # L2 Regularization Penalty
L1L2(l1=0.01, l2=0.01)  # L1 + L2 penalties
```

## Directly calling a regularizer

Compute a regularization loss on a tensor by directly calling a regularizer as if it is a one-argument function.

E.g.

```console
>>> regularizer = L2(2.)
>>> tensor = ops.ones(shape=(5, 5))
>>> regularizer(tensor)
50.0
```

## Developing new regularizers

Any function that takes in a weight matrix and returns a scalar tensor can be used as a regularizer, e.g.:

```console
>>> def l1_reg(weight_matrix):
...    return 0.01 * ops.sum(ops.absolute(weight_matrix))
...
>>> layer = Dense(5, input_dim=5,
...     kernel_initializer='ones', kernel_regularizer=l1_reg)
>>> tensor = ops.ones(shape=(5, 5))
>>> out = layer(tensor)
>>> layer.losses
0.25
```

Alternatively, you can write your custom regularizers in an object-oriented way by extending this regularizer base class, e.g.:

```console
>>> class L2Regularizer(Regularizer):
...   def __init__(self, l2=0.):
...     self.l2 = l2
...
...   def __call__(self, x):
...     return self.l2 * ops.sum(ops.square(x))
...
...   def get_config(self):
...     return {'l2': float(self.l2)}
...
>>> layer = Dense(
...   5, input_dim=5, kernel_initializer='ones',
...   kernel_regularizer=L2Regularizer(l2=0.5))
```

```console
>>> tensor = ops.ones(shape=(5, 5))
>>> out = layer(tensor)
>>> layer.losses
12.5
```

### A note on serialization and deserialization:

Registering the regularizers as serializable is optional if you are just training and executing models, exporting to and from SavedModels, or saving and loading weight checkpoints.

Registration is required for saving and loading models to HDF5 format, Keras model cloning, some visualization utilities, and exporting models to and from JSON. If using this functionality, you must make sure any python process running your model has also defined and registered your custom regularizer.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/regularizers/regularizers.py#L213" >}}

### `L1` class

```python
keras.regularizers.L1(l1=0.01)
```

A regularizer that applies a L1 regularization penalty.

The L1 regularization penalty is computed as: `loss = l1 * reduce_sum(abs(x))`

L1 may be passed to a layer as a string identifier:

```console
>>> dense = Dense(3, kernel_regularizer='l1')
```

In this case, the default value used is `l1=0.01`.

**Arguments**

- **l1**: float, L1 regularization factor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/regularizers/regularizers.py#L242" >}}

### `L2` class

```python
keras.regularizers.L2(l2=0.01)
```

A regularizer that applies a L2 regularization penalty.

The L2 regularization penalty is computed as: `loss = l2 * reduce_sum(square(x))`

L2 may be passed to a layer as a string identifier:

```console
>>> dense = Dense(3, kernel_regularizer='l2')
```

In this case, the default value used is `l2=0.01`.

**Arguments**

- **l2**: float, L2 regularization factor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/regularizers/regularizers.py#L168" >}}

### `L1L2` class

```python
keras.regularizers.L1L2(l1=0.0, l2=0.0)
```

A regularizer that applies both L1 and L2 regularization penalties.

The L1 regularization penalty is computed as: `loss = l1 * reduce_sum(abs(x))`

The L2 regularization penalty is computed as `loss = l2 * reduce_sum(square(x))`

L1L2 may be passed to a layer as a string identifier:

```console
>>> dense = Dense(3, kernel_regularizer='l1_l2')
```

In this case, the default values used are `l1=0.01` and `l2=0.01`.

**Arguments**

- **l1**: float, L1 regularization factor.
- **l2**: float, L2 regularization factor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/regularizers/regularizers.py#L271" >}}

### `OrthogonalRegularizer` class

```python
keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="rows")
```

Regularizer that encourages input vectors to be orthogonal to each other.

It can be applied to either the rows of a matrix (`mode="rows"`) or its columns (`mode="columns"`). When applied to a `Dense` kernel of shape `(input_dim, units)`, rows mode will seek to make the feature vectors (i.e. the basis of the output space) orthogonal to each other.

**Arguments**

- **factor**: Float. The regularization factor. The regularization penalty will be proportional to `factor` times the mean of the dot products between the L2-normalized rows (if `mode="rows"`, or columns if `mode="columns"`) of the inputs, excluding the product of each row/column with itself. Defaults to `0.01`.
- **mode**: String, one of `{"rows", "columns"}`. Defaults to `"rows"`. In rows mode, the regularization effect seeks to make the rows of the input orthogonal to each other. In columns mode, it seeks to make the columns of the input orthogonal to each other.

**Example**

```console
>>> regularizer = OrthogonalRegularizer(factor=0.01)
>>> layer = Dense(units=4, kernel_regularizer=regularizer)
```

## Creating custom regularizers

### Simple callables

A weight regularizer can be any callable that takes as input a weight tensor (e.g. the kernel of a `Conv2D` layer), and returns a scalar loss. Like this:

```python
def my_regularizer(x):
    return 1e-3 * ops.sum(ops.square(x))
```

### `Regularizer` subclasses

If you need to configure your regularizer via various arguments (e.g. `l1` and `l2` arguments in `l1_l2`), you should implement it as a subclass of [`keras.regularizers.Regularizer`]({{< relref "/docs/api/layers/regularizers#regularizer-class" >}}).

Here's a simple example:

```python
class MyRegularizer(regularizers.Regularizer):

    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        return self.strength * ops.sum(ops.square(x))
```

Optionally, you can also implement the method `get_config` and the class method `from_config` in order to support serialization – just like with any Keras object. Example:

```python
class MyRegularizer(regularizers.Regularizer):

    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        return self.strength * ops.sum(ops.square(x))

    def get_config(self):
        return {'strength': self.strength}
```
