---
title: Mixed precision policy API
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/dtype_policies/dtype_policy.py#L9" >}}

### `DTypePolicy` class

```python
keras.dtype_policies.DTypePolicy(name=None)
```

A dtype policy for a Keras layer.

A dtype policy determines a layer's computation and variable dtypes. Each
layer has a policy. Policies can be passed to the `dtype` argument of layer
constructors, or a global policy can be set with
[`keras.config.set_dtype_policy`]({{< relref "/docs/api/mixed_precision/policy#set_dtype_policy-function" >}}).

**Arguments**

- **name**: The policy name, which determines the compute and variable dtypes.
  Can be any dtype name, such as `"float32"` or `"float64"`,
  which causes both the compute and variable dtypes
  will be that dtype.
  Can also be the string `"mixed_float16"` or `"mixed_bfloat16"`,
  which causes the compute dtype to be `float16` or `bfloat16`
  and the variable dtype to be `float32`.

Typically you only need to interact with dtype policies when using mixed
precision, which is the use of float16 or bfloat16 for computations and
float32 for variables. This is why the term `mixed_precision` appears in the
API name. Mixed precision can be enabled by passing `"mixed_float16"` or
`"mixed_bfloat16"` to `keras.mixed_precision.set_dtype_policy()`.

```console
>>> keras.config.set_dtype_policy("mixed_float16")
>>> layer1 = keras.layers.Dense(10)
>>> layer1.dtype_policy  # layer1 will automatically use mixed precision
<DTypePolicy "mixed_float16">
>>> # Can optionally override layer to use float32
>>> # instead of mixed precision.
>>> layer2 = keras.layers.Dense(10, dtype="float32")
>>> layer2.dtype_policy
<DTypePolicy "float32">
>>> # Set policy back to initial float32.
>>> keras.config.set_dtype_policy('float32')
```

In the example above, passing `dtype="float32"` to the layer is
equivalent to passing
`dtype=keras.config.DTypePolicy("float32")`.
In general, passing a dtype policy name to a layer is equivalent
to passing the corresponding policy, so it is never necessary
to explicitly construct a `DTypePolicy` object.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/dtype_policies/dtype_policy_map.py#L9" >}}

### `DTypePolicyMap` class

```python
keras.dtype_policies.DTypePolicyMap(default_policy=None, policy_map=None)
```

Dict-like object mapping layer paths to `DTypePolicy` instances.

`DTypePolicyMap` can be used in `get_config` in layers and subclasses to
support a complex configurations of dtype policies.

For example, we can modify `get_config` in `layers.MultiHeadAttention` as
follows to support the mixing of dtype policies, such as quantization.

```python
@keras.saving.register_keras_serializable("MyPackage")
class MyMultiHeadAttention(keras.layers.MultiHeadAttention):
    def get_config(self):
        config = super().get_config()
        dtype_policy_map = dtype_policies.DTypePolicyMap()
        for layer in self._flatten_layers():
            if layer.dtype_policy.quantization_mode is not None:
                dtype_policy_map[layer.path] = layer.dtype_policy
        if len(dtype_policy_map) > 0:
            config.update({"dtype": dtype_policy_map})
        return config
```

Internally, `DTypePolicyMap` uses a string as a key and a `DTypePolicy`
as the value. Typically, the key used for querying is the `Layer.path`.
However, it is also possible to set a regex as the key. See the docstring of
`get` for more details.

See below for a usage example. You can define the naming schema
of the `DTypePolicy`, and then retrieve the corresponding `DTypePolicy`
instance.

```python
dtype_policy_map = DTypePolicyMap()
dtype_policy_map["layer/dense_0"] = DTypePolicy("bfloat16")
dtype_policy_map["layer/dense_1"] = QuantizedDTypePolicy("int8", "bfloat16")
policy_0 = dtype_policy_map["layer/dense_0"]
policy_1 = dtype_policy_map["layer/dense_1"]
policy_2 = dtype_policy_map["layer/dense_2"]  # No hit
assert policy_0 == DTypePolicy("bfloat16")
assert policy_1 == QuantizedDTypePolicy("int8", "bfloat16")
assert policy_2 == keras.config.dtype_policy()
```

**Arguments**

- **default_policy**: An optional `DTypePolicy` instance specifying the
  default dtype policy. If not specified, the value will default to
  `keras.config.dtype_policy()`.
- **policy_map**: An optional dict that maps string to `DTypePolicy`
  instances. Defaults to `None`

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/dtype_policies/dtype_policy.py#L207" >}}

### `FloatDTypePolicy` class

```python
keras.dtype_policies.FloatDTypePolicy(name=None)
```

A dtype policy for a Keras layer.

A dtype policy determines a layer's computation and variable dtypes. Each
layer has a policy. Policies can be passed to the `dtype` argument of layer
constructors, or a global policy can be set with
[`keras.config.set_dtype_policy`]({{< relref "/docs/api/mixed_precision/policy#set_dtype_policy-function" >}}).

**Arguments**

- **name**: The policy name, which determines the compute and variable dtypes.
  Can be any dtype name, such as `"float32"` or `"float64"`,
  which causes both the compute and variable dtypes
  will be that dtype.
  Can also be the string `"mixed_float16"` or `"mixed_bfloat16"`,
  which causes the compute dtype to be `float16` or `bfloat16`
  and the variable dtype to be `float32`.

Typically you only need to interact with dtype policies when using mixed
precision, which is the use of float16 or bfloat16 for computations and
float32 for variables. This is why the term `mixed_precision` appears in the
API name. Mixed precision can be enabled by passing `"mixed_float16"` or
`"mixed_bfloat16"` to `keras.mixed_precision.set_dtype_policy()`.

```console
>>> keras.config.set_dtype_policy("mixed_float16")
>>> layer1 = keras.layers.Dense(10)
>>> layer1.dtype_policy  # layer1 will automatically use mixed precision
<DTypePolicy "mixed_float16">
>>> # Can optionally override layer to use float32
>>> # instead of mixed precision.
>>> layer2 = keras.layers.Dense(10, dtype="float32")
>>> layer2.dtype_policy
<DTypePolicy "float32">
>>> # Set policy back to initial float32.
>>> keras.config.set_dtype_policy('float32')
```

In the example above, passing `dtype="float32"` to the layer is
equivalent to passing
`dtype=keras.config.DTypePolicy("float32")`.
In general, passing a dtype policy name to a layer is equivalent
to passing the corresponding policy, so it is never necessary
to explicitly construct a `DTypePolicy` object.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/dtype_policies/dtype_policy.py#L215" >}}

### `QuantizedDTypePolicy` class

```python
keras.dtype_policies.QuantizedDTypePolicy(mode, source_name=None)
```

A dtype policy for a Keras layer.

A dtype policy determines a layer's computation and variable dtypes. Each
layer has a policy. Policies can be passed to the `dtype` argument of layer
constructors, or a global policy can be set with
[`keras.config.set_dtype_policy`]({{< relref "/docs/api/mixed_precision/policy#set_dtype_policy-function" >}}).

**Arguments**

- **name**: The policy name, which determines the compute and variable dtypes.
  Can be any dtype name, such as `"float32"` or `"float64"`,
  which causes both the compute and variable dtypes
  will be that dtype.
  Can also be the string `"mixed_float16"` or `"mixed_bfloat16"`,
  which causes the compute dtype to be `float16` or `bfloat16`
  and the variable dtype to be `float32`.

Typically you only need to interact with dtype policies when using mixed
precision, which is the use of float16 or bfloat16 for computations and
float32 for variables. This is why the term `mixed_precision` appears in the
API name. Mixed precision can be enabled by passing `"mixed_float16"` or
`"mixed_bfloat16"` to `keras.mixed_precision.set_dtype_policy()`.

```console
>>> keras.config.set_dtype_policy("mixed_float16")
>>> layer1 = keras.layers.Dense(10)
>>> layer1.dtype_policy  # layer1 will automatically use mixed precision
<DTypePolicy "mixed_float16">
>>> # Can optionally override layer to use float32
>>> # instead of mixed precision.
>>> layer2 = keras.layers.Dense(10, dtype="float32")
>>> layer2.dtype_policy
<DTypePolicy "float32">
>>> # Set policy back to initial float32.
>>> keras.config.set_dtype_policy('float32')
```

In the example above, passing `dtype="float32"` to the layer is
equivalent to passing
`dtype=keras.config.DTypePolicy("float32")`.
In general, passing a dtype policy name to a layer is equivalent
to passing the corresponding policy, so it is never necessary
to explicitly construct a `DTypePolicy` object.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/dtype_policies/dtype_policy.py#L259" >}}

### `QuantizedFloat8DTypePolicy` class

```python
keras.dtype_policies.QuantizedFloat8DTypePolicy(
    mode, source_name=None, amax_history_length=1024
)
```

A dtype policy for a Keras layer.

A dtype policy determines a layer's computation and variable dtypes. Each
layer has a policy. Policies can be passed to the `dtype` argument of layer
constructors, or a global policy can be set with
[`keras.config.set_dtype_policy`]({{< relref "/docs/api/mixed_precision/policy#set_dtype_policy-function" >}}).

**Arguments**

- **name**: The policy name, which determines the compute and variable dtypes.
  Can be any dtype name, such as `"float32"` or `"float64"`,
  which causes both the compute and variable dtypes
  will be that dtype.
  Can also be the string `"mixed_float16"` or `"mixed_bfloat16"`,
  which causes the compute dtype to be `float16` or `bfloat16`
  and the variable dtype to be `float32`.

Typically you only need to interact with dtype policies when using mixed
precision, which is the use of float16 or bfloat16 for computations and
float32 for variables. This is why the term `mixed_precision` appears in the
API name. Mixed precision can be enabled by passing `"mixed_float16"` or
`"mixed_bfloat16"` to `keras.mixed_precision.set_dtype_policy()`.

```console
>>> keras.config.set_dtype_policy("mixed_float16")
>>> layer1 = keras.layers.Dense(10)
>>> layer1.dtype_policy  # layer1 will automatically use mixed precision
<DTypePolicy "mixed_float16">
>>> # Can optionally override layer to use float32
>>> # instead of mixed precision.
>>> layer2 = keras.layers.Dense(10, dtype="float32")
>>> layer2.dtype_policy
<DTypePolicy "float32">
>>> # Set policy back to initial float32.
>>> keras.config.set_dtype_policy('float32')
```

In the example above, passing `dtype="float32"` to the layer is
equivalent to passing
`dtype=keras.config.DTypePolicy("float32")`.
In general, passing a dtype policy name to a layer is equivalent
to passing the corresponding policy, so it is never necessary
to explicitly construct a `DTypePolicy` object.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/dtype_policies/dtype_policy.py#L322" >}}

### `dtype_policy` function

```python
keras.config.dtype_policy()
```

Returns the current default dtype policy object.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/dtype_policies/dtype_policy.py#L291" >}}

### `set_dtype_policy` function

```python
keras.config.set_dtype_policy(policy)
```

Sets the default dtype policy globally.

**Example**

```console
>>> keras.config.set_dtype_policy("mixed_float16")
```
