---
title: Model config serialization
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/layer.py#L1531" >}}

### `get_config` method

```python
Model.get_config()
```

Returns the config of the object.

An object config is a Python dictionary (serializable) containing the information needed to re-instantiate it.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L499" >}}

### `from_config` method

```python
Model.from_config(config, custom_objects=None)
```

Creates an operation from its config.

This method is the reverse of `get_config`, capable of instantiating the same operation from the config dictionary.

Note: If you override this method, you might receive a serialized dtype config, which is a `dict`. You can deserialize it as follows:

```python
if "dtype" in config and isinstance(config["dtype"], dict):
    policy = dtype_policies.deserialize(config["dtype"])
```

**Arguments**

- **config**: A Python dictionary, typically the output of `get_config`.

**Returns**

An operation instance.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/cloning.py#L13" >}}

### `clone_model` function

```python
keras.models.clone_model(
    model,
    input_tensors=None,
    clone_function=None,
    call_function=None,
    recursive=False,
    **kwargs
)
```

Clone a Functional or Sequential `Model` instance.

Model cloning is similar to calling a model on new inputs, except that it creates new layers (and thus new weights) instead of sharing the weights of the existing layers.

Note that `clone_model` will not preserve the uniqueness of shared objects within the model (e.g. a single variable attached to two distinct layers will be restored as two separate variables).

**Arguments**

- **model**: Instance of `Model` (could be a Functional model or a Sequential model).
- **input_tensors**: optional list of input tensors or InputLayer objects to build the model upon. If not provided, new `Input` objects will be created.
- **clone_function**: Callable with signature `fn(layer)` to be used to clone each layer in the target model (except `Input` instances). It takes as argument the layer instance to be cloned, and returns the corresponding layer instance to be used in the model copy. If unspecified, this callable defaults to the following serialization/deserialization function: `lambda layer: layer.__class__.from_config(layer.get_config())`. By passing a custom callable, you can customize your copy of the model, e.g. by wrapping certain layers of interest (you might want to replace all `LSTM` instances with equivalent `Bidirectional(LSTM(...))` instances, for example). Defaults to `None`.
- **call_function**: Callable with signature `fn(layer, *args, **kwargs)` to be used to call each cloned layer and a set of inputs. It takes the layer instance, the call arguments and keyword arguments, and returns the call outputs. If unspecified, this callable defaults to the regular `__call__()` method: `def fn(layer, *args, **kwargs): return layer(*args, **kwargs)`. By passing a custom callable, you can insert new layers before or after a given layer. Note: this argument can only be used with Functional models.
- **recursive**: Boolean. Whether to recursively clone any Sequential or Functional models encountered in the original Sequential/Functional model. If `False`, then inner models are cloned by calling `clone_function()`. If `True`, then inner models are cloned by calling `clone_model()` with the same `clone_function`, `call_function`, and `recursive` arguments. Note that in this case, `call_function` will not be propagated to any Sequential model (since it is not applicable to Sequential models).

**Returns**

An instance of `Model` reproducing the behavior of the original model, on top of new inputs tensors, using newly instantiated weights. The cloned model may behave differently from the original model if a custom `clone_function` or `call_function` modifies a layer or layer call.

**Example**

```python
# Create a test Sequential model.
model = keras.Sequential([
    keras.layers.Input(shape=(728,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
# Create a copy of the test model (with freshly initialized weights).
new_model = clone_model(model)
```

Using a `clone_function` to make a model deterministic by setting the random seed everywhere:

```python
def clone_function(layer):
    config = layer.get_config()
    if "seed" in config:
        config["seed"] = 1337
    return layer.__class__.from_config(config)

new_model = clone_model(model, clone_function=clone_function)
```

Using a `call_function` to add a `Dropout` layer after each `Dense` layer (without recreating new layers):

```python
def call_function(layer, *args, **kwargs):
    out = layer(*args, **kwargs)
    if isinstance(layer, keras.layers.Dense):
        out = keras.layers.Dropout(0.5)(out)
    return out

new_model = clone_model(
    model,
    clone_function=lambda x: x,  # Reuse the same layers.
    call_function=call_function,
)
```

Note that subclassed models cannot be cloned by default, since their internal layer structure is not known. To achieve equivalent functionality as `clone_model` in the case of a subclassed model, simply make sure that the model class implements `get_config()` (and optionally `from_config()`), and call:

```python
new_model = model.__class__.from_config(model.get_config())
```

In the case of a subclassed model, you cannot using a custom `clone_function`.
