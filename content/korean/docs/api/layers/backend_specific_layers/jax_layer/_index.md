---
title: JaxLayer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/jax_layer.py#L15" >}}

### `JaxLayer` class

```python
keras.layers.JaxLayer(
    call_fn, init_fn=None, params=None, state=None, seed=None, **kwargs
)
```

Keras Layer that wraps a JAX model.

This layer enables the use of JAX components within Keras when using JAX as
the backend for Keras.

## Model function

This layer accepts JAX models in the form of a function, `call_fn`, which
must take the following arguments with these exact names:

- `params`: trainable parameters of the model.
- `state` (_optional_): non-trainable state of the model. Can be omitted if
  the model has no non-trainable state.
- `rng` (_optional_): a `jax.random.PRNGKey` instance. Can be omitted if the
  model does not need RNGs, neither during training nor during inference.
- `inputs`: inputs to the model, a JAX array or a `PyTree` of arrays.
- `training` (_optional_): an argument specifying if we're in training mode
  or inference mode, `True` is passed in training mode. Can be omitted if
  the model behaves the same in training mode and inference mode.

The `inputs` argument is mandatory. Inputs to the model must be provided via
a single argument. If the JAX model takes multiple inputs as separate
arguments, they must be combined into a single structure, for instance in a
`tuple` or a `dict`.

## Model weights initialization

The initialization of the `params` and `state` of the model can be handled
by this layer, in which case the `init_fn` argument must be provided. This
allows the model to be initialized dynamically with the right shape.
Alternatively, and if the shape is known, the `params` argument and
optionally the `state` argument can be used to create an already initialized
model.

The `init_fn` function, if provided, must take the following arguments with
these exact names:

- `rng`: a `jax.random.PRNGKey` instance.
- `inputs`: a JAX array or a `PyTree` of arrays with placeholder values to
  provide the shape of the inputs.
- `training` (_optional_): an argument specifying if we're in training mode
  or inference mode. `True` is always passed to `init_fn`. Can be omitted
  regardless of whether `call_fn` has a `training` argument.

## Models with non-trainable state

For JAX models that have non-trainable state:

- `call_fn` must have a `state` argument
- `call_fn` must return a `tuple` containing the outputs of the model and
  the new non-trainable state of the model
- `init_fn` must return a `tuple` containing the initial trainable params of
  the model and the initial non-trainable state of the model.

This code shows a possible combination of `call_fn` and `init_fn` signatures
for a model with non-trainable state. In this example, the model has a
`training` argument and an `rng` argument in `call_fn`.

```python
def stateful_call(params, state, rng, inputs, training):
    outputs = ...
    new_state = ...
    return outputs, new_state
def stateful_init(rng, inputs):
    initial_params = ...
    initial_state = ...
    return initial_params, initial_state
```

## Models without non-trainable state

For JAX models with no non-trainable state:

- `call_fn` must not have a `state` argument
- `call_fn` must return only the outputs of the model
- `init_fn` must return only the initial trainable params of the model.

This code shows a possible combination of `call_fn` and `init_fn` signatures
for a model without non-trainable state. In this example, the model does not
have a `training` argument and does not have an `rng` argument in `call_fn`.

```python
def stateless_call(params, inputs):
    outputs = ...
    return outputs
def stateless_init(rng, inputs):
    initial_params = ...
    return initial_params
```

## Conforming to the required signature

If a model has a different signature than the one required by `JaxLayer`,
one can easily write a wrapper method to adapt the arguments. This example
shows a model that has multiple inputs as separate arguments, expects
multiple RNGs in a `dict`, and has a `deterministic` argument with the
opposite meaning of `training`. To conform, the inputs are combined in a
single structure using a `tuple`, the RNG is split and used the populate the
expected `dict`, and the Boolean flag is negated:

```python
def my_model_fn(params, rngs, input1, input2, deterministic):
    ...
    if not deterministic:
        dropout_rng = rngs["dropout"]
        keep = jax.random.bernoulli(dropout_rng, dropout_rate, x.shape)
        x = jax.numpy.where(keep, x / dropout_rate, 0)
        ...
    ...
    return outputs
def my_model_wrapper_fn(params, rng, inputs, training):
    input1, input2 = inputs
    rng1, rng2 = jax.random.split(rng)
    rngs = {"dropout": rng1, "preprocessing": rng2}
    deterministic = not training
    return my_model_fn(params, rngs, input1, input2, deterministic)
keras_layer = JaxLayer(my_model_wrapper_fn, params=initial_params)
```

## Usage with Haiku modules

`JaxLayer` enables the use of [Haiku](https://dm-haiku.readthedocs.io)
components in the form of
[`haiku.Module`](https://dm-haiku.readthedocs.io/en/latest/api.html#module).
This is achieved by transforming the module per the Haiku pattern and then
passing `module.apply` in the `call_fn` parameter and `module.init` in the
`init_fn` parameter if needed.

If the model has non-trainable state, it should be transformed with
[`haiku.transform_with_state`](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform_with_state).
If the model has no non-trainable state, it should be transformed with
[`haiku.transform`](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform).
Additionally, and optionally, if the module does not use RNGs in "apply", it
can be transformed with
[`haiku.without_apply_rng`](https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng).

The following example shows how to create a `JaxLayer` from a Haiku module
that uses random number generators via `hk.next_rng_key()` and takes a
training positional argument:

```python
class MyHaikuModule(hk.Module):
    def __call__(self, x, training):
        x = hk.Conv2D(32, (3, 3))(x)
        x = jax.nn.relu(x)
        x = hk.AvgPool((1, 2, 2, 1), (1, 2, 2, 1), "VALID")(x)
        x = hk.Flatten()(x)
        x = hk.Linear(200)(x)
        if training:
            x = hk.dropout(rng=hk.next_rng_key(), rate=0.3, x=x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        x = jax.nn.softmax(x)
        return x
def my_haiku_module_fn(inputs, training):
    module = MyHaikuModule()
    return module(inputs, training)
transformed_module = hk.transform(my_haiku_module_fn)
keras_layer = JaxLayer(
    call_fn=transformed_module.apply,
    init_fn=transformed_module.init,
)
```

**Arguments**

- **call_fn**: The function to call the model. See description above for the
  list of arguments it takes and the outputs it returns.
  init_fn: the function to call to initialize the model. See description
  above for the list of arguments it takes and the ouputs it returns.
  If `None`, then `params` and/or `state` must be provided.
- **params**: A `PyTree` containing all the model trainable parameters. This
  allows passing trained parameters or controlling the initialization.
  If both `params` and `state` are `None`, `init_fn` is called at
  build time to initialize the trainable parameters of the model.
- **state**: A `PyTree` containing all the model non-trainable state. This
  allows passing learned state or controlling the initialization. If
  both `params` and `state` are `None`, and `call_fn` takes a `state`
  argument, then `init_fn` is called at build time to initialize the
  non-trainable state of the model.
- **seed**: Seed for random number generator. Optional.
