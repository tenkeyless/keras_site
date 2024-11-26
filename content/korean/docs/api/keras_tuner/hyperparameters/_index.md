---
title: HyperParameters
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L29" >}}

### `HyperParameters` class

```python
keras_tuner.HyperParameters()
```

Container for both a hyperparameter space, and current values.

A `HyperParameters` instance can be pass to `HyperModel.build(hp)` as an
argument to build a model.

To prevent the users from depending on inactive hyperparameter values, only
active hyperparameters should have values in `HyperParameters.values`.

**Attributes**

- **space**: A list of `HyperParameter` objects.
- **values**: A dict mapping hyperparameter names to current values.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L496" >}}

### `Boolean` method

```python
HyperParameters.Boolean(name, default=False, parent_name=None, parent_values=None)
```

Choice between True and False.

**Arguments**

- **name**: A string. the name of parameter. Must be unique for each
  `HyperParameter` instance in the search space.
- **default**: Boolean, the default value to return for the parameter.
  If unspecified, the default value will be False.
- **parent_name**: Optional string, specifying the name of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.
- **parent_values**: Optional list of the values of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.

**Returns**

The value of the hyperparameter, or None if the hyperparameter is
not active.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L258" >}}

### `Choice` method

```python
HyperParameters.Choice(
    name, values, ordered=None, default=None, parent_name=None, parent_values=None
)
```

Choice of one value among a predefined set of possible values.

**Arguments**

- **name**: A string. the name of parameter. Must be unique for each
  `HyperParameter` instance in the search space.
- **values**: A list of possible values. Values must be int, float,
  str, or bool. All values must be of the same type.
- **ordered**: Optional boolean, whether the values passed should be
  considered to have an ordering. Defaults to `True` for float/int
  values. Must be `False` for any other values.
- **default**: Optional default value to return for the parameter.
  If unspecified, the default value will be:
  - None if None is one of the choices in `values`
  - The first entry in `values` otherwise.
- **parent_name**: Optional string, specifying the name of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.
- **parent_values**: Optional list of the values of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.

**Returns**

The value of the hyperparameter, or None if the hyperparameter is
not active.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L525" >}}

### `Fixed` method

```python
HyperParameters.Fixed(name, value, parent_name=None, parent_values=None)
```

Fixed, untunable value.

**Arguments**

- **name**: A string. the name of parameter. Must be unique for each
  `HyperParameter` instance in the search space.
- **value**: The value to use (can be any JSON-serializable Python type).
- **parent_name**: Optional string, specifying the name of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.
- **parent_values**: Optional list of the values of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.

**Returns**

The value of the hyperparameter, or None if the hyperparameter is
not active.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L401" >}}

### `Float` method

```python
HyperParameters.Float(
    name,
    min_value,
    max_value,
    step=None,
    sampling="linear",
    default=None,
    parent_name=None,
    parent_values=None,
)
```

Floating point value hyperparameter.

Example #1:

```python
hp.Float(
    "image_rotation_factor",
    min_value=0,
    max_value=1)
```

All values in interval [0, 1] have equal probability of being sampled.

Example #2:

```python
hp.Float(
    "image_rotation_factor",
    min_value=0,
    max_value=1,
    step=0.2)
```

`step` is the minimum distance between samples.
The possible values are [0, 0.2, 0.4, 0.6, 0.8, 1.0].

Example #3:

```python
hp.Float(
    "learning_rate",
    min_value=0.001,
    max_value=10,
    step=10,
    sampling="log")
```

When `sampling="log"`, the `step` is multiplied between samples.
The possible values are [0.001, 0.01, 0.1, 1, 10].

**Arguments**

- **name**: A string. the name of parameter. Must be unique for each
  `HyperParameter` instance in the search space.
- **min_value**: Float, the lower bound of the range.
- **max_value**: Float, the upper bound of the range.
- **step**: Optional float, the distance between two consecutive samples
  in the range. If left unspecified, it is possible to sample any
  value in the interval. If `sampling="linear"`, it will be the
  minimum additve between two samples. If `sampling="log"`, it
  will be the minimum multiplier between two samples.
- **sampling**: String. One of "linear", "log", "reverse_log". Defaults to
  "linear". When sampling value, it always start from a value in
  range [0.0, 1.0). The `sampling` argument decides how the value
  is projected into the range of [min\_value, max\_value].
  "linear": min_value + value \* (max_value - min_value)
  "log": min_value \* (max_value / min_value) ^ value
  "reverse_log":
  (max_value -
  min_value \* ((max_value / min_value) ^ (1 - value) - 1))
- **default**: Float, the default value to return for the parameter. If
  unspecified, the default value will be `min_value`.
- **parent_name**: Optional string, specifying the name of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.
- **parent_values**: Optional list of the values of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.

**Returns**

The value of the hyperparameter, or None if the hyperparameter is
not active.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L302" >}}

### `Int` method

```python
HyperParameters.Int(
    name,
    min_value,
    max_value,
    step=None,
    sampling="linear",
    default=None,
    parent_name=None,
    parent_values=None,
)
```

Integer hyperparameter.

Note that unlike Python's `range` function, `max_value` is _included_ in
the possible values this parameter can take on.

Example #1:

```python
hp.Int(
    "n_layers",
    min_value=6,
    max_value=12)
```

The possible values are [6, 7, 8, 9, 10, 11, 12].

Example #2:

```python
hp.Int(
    "n_layers",
    min_value=6,
    max_value=13,
    step=3)
```

`step` is the minimum distance between samples.
The possible values are [6, 9, 12].

Example #3:

```python
hp.Int(
    "batch_size",
    min_value=2,
    max_value=32,
    step=2,
    sampling="log")
```

When `sampling="log"` the `step` is multiplied between samples.
The possible values are [2, 4, 8, 16, 32].

**Arguments**

- **name**: A string. the name of parameter. Must be unique for each
  `HyperParameter` instance in the search space.
- **min_value**: Integer, the lower limit of range, inclusive.
- **max_value**: Integer, the upper limit of range, inclusive.
- **step**: Optional integer, the distance between two consecutive samples
  in the range. If left unspecified, it is possible to sample any
  integers in the interval. If `sampling="linear"`, it will be the
  minimum additve between two samples. If `sampling="log"`, it
  will be the minimum multiplier between two samples.
- **sampling**: String. One of "linear", "log", "reverse_log". Defaults to
  "linear". When sampling value, it always start from a value in
  range [0.0, 1.0). The `sampling` argument decides how the value
  is projected into the range of [min\_value, max\_value].
  "linear": min_value + value \* (max_value - min_value)
  "log": min_value \* (max_value / min_value) ^ value
  "reverse_log":
  (max_value -
  min_value \* ((max_value / min_value) ^ (1 - value) - 1))
- **default**: Integer, default value to return for the parameter. If
  unspecified, the default value will be `min_value`.
- **parent_name**: Optional string, specifying the name of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.
- **parent_values**: Optional list of the values of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.

**Returns**

The value of the hyperparameter, or None if the hyperparameter is
not active.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L87" >}}

### `conditional_scope` method

```python
HyperParameters.conditional_scope(parent_name, parent_values)
```

Opens a scope to create conditional HyperParameters.

All `HyperParameter`s created under this scope will only be active when
the parent `HyperParameter` specified by `parent_name` is equal to one
of the values passed in `parent_values`.

When the condition is not met, creating a `HyperParameter` under this
scope will register the `HyperParameter`, but will return `None` rather
than a concrete value.

Note that any Python code under this scope will execute regardless of
whether the condition is met.

This feature is for the `Tuner` to collect more information of the
search space and the current trial. It is especially useful for model
selection. If the parent `HyperParameter` is for model selection, the
`HyperParameter`s in a model should only be active when the model
selected, which can be implemented using `conditional_scope`.

**Examples**

```python
def MyHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(32, 32, 3)))
        model_type = hp.Choice("model_type", ["mlp", "cnn"])
        with hp.conditional_scope("model_type", ["mlp"]):
            if model_type == "mlp":
                model.add(Flatten())
                model.add(Dense(32, activation='relu'))
        with hp.conditional_scope("model_type", ["cnn"]):
            if model_type == "cnn":
                model.add(Conv2D(64, 3, activation='relu'))
                model.add(GlobalAveragePooling2D())
        model.add(Dense(10, activation='softmax'))
        return model
```

**Arguments**

- **parent_name**: A string, specifying the name of the parent
  `HyperParameter` to use as the condition to activate the
  current `HyperParameter`.
- **parent_values**: A list of the values of the parent `HyperParameter`
  to use as the condition to activate the current
  `HyperParameter`.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hyperparameters/hyperparameters.py#L238" >}}

### `get` method

```python
HyperParameters.get(name)
```

Return the current value of this hyperparameter set.
