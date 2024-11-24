---
title: The Sequential class
toc: true
weight: 2
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/sequential.py#L19" >}}

### `Sequential` class

```python
keras.Sequential(layers=None, trainable=True, name=None)
```

`Sequential` groups a linear stack of layers into a `Model`.

**Examples**

```python
model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))

# Note that you can also omit the initial `Input`.
# In that case the model doesn't have any weights until the first call
# to a training/evaluation method (since it isn't yet built):
model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
# model.weights not created yet

# Whereas if you specify an `Input`, the model gets built
# continuously as you are adding layers:
model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
len(model.weights)  # Returns "2"

# When using the delayed-build pattern (no input shape specified), you can
# choose to manually build your model by calling
# `build(batch_input_shape)`:
model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
model.build((None, 16))
len(model.weights)  # Returns "4"

# Note that when using the delayed-build pattern (no input shape specified),
# the model gets built the first time you call `fit`, `eval`, or `predict`,
# or the first time you call the model on some input data.
model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(1))
model.compile(optimizer='sgd', loss='mse')
# This builds the model for the first time:
model.fit(x, y, batch_size=32, epochs=10)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/sequential.py#L78" >}}

### `add` method

```python
Sequential.add(layer, rebuild=True)
```

Adds a layer instance on top of the layer stack.

**Arguments**

- **layer**: layer instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/sequential.py#L127" >}}

### `pop` method

```python
Sequential.pop(rebuild=True)
```

Removes the last layer in the model.
