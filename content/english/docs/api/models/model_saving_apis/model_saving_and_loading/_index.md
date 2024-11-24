---
title: Whole model saving & loading
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L267" >}}

### `save` method

```python
Model.save(filepath, overwrite=True, zipped=None, **kwargs)
```

Saves a model as a `.keras` file.

**Arguments**

- **filepath**: `str` or `pathlib.Path` object. The path where to save the model. Must end in `.keras` (unless saving the model as an unzipped directory via `zipped=False`).
- **overwrite**: Whether we should overwrite any existing model at the target location, or instead ask the user via an interactive prompt.
- **zipped**: Whether to save the model as a zipped `.keras` archive (default when saving locally), or as an unzipped directory (default when saving on the Hugging Face Hub).

**Example**

```python
model = keras.Sequential(
    [
        keras.layers.Dense(5, input_shape=(3,)),
        keras.layers.Softmax(),
    ],
)
model.save("model.keras")
loaded_model = keras.saving.load_model("model.keras")
x = keras.random.uniform((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))
```

Note that `model.save()` is an alias for `keras.saving.save_model()`.

The saved `.keras` file contains:

- The model's configuration (architecture)
- The model's weights
- The model's optimizer's state (if any)

Thus models can be reinstantiated in the exact same state.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/saving_api.py#L18" >}}

### `save_model` function

```python
keras.saving.save_model(model, filepath, overwrite=True, zipped=None, **kwargs)
```

Saves a model as a `.keras` file.

**Arguments**

- **model**: Keras model instance to be saved.
- **filepath**: `str` or `pathlib.Path` object. Path where to save the model.
- **overwrite**: Whether we should overwrite any existing model at the target location, or instead ask the user via an interactive prompt.
- **zipped**: Whether to save the model as a zipped `.keras` archive (default when saving locally), or as an unzipped directory (default when saving on the Hugging Face Hub).

**Example**

```python
model = keras.Sequential(
    [
        keras.layers.Dense(5, input_shape=(3,)),
        keras.layers.Softmax(),
    ],
)
model.save("model.keras")
loaded_model = keras.saving.load_model("model.keras")
x = keras.random.uniform((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))
```

Note that `model.save()` is an alias for `keras.saving.save_model()`.

The saved `.keras` file is a `zip` archive that contains:

- The model's configuration (architecture)
- The model's weights
- The model's optimizer's state (if any)

Thus models can be reinstantiated in the exact same state.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/saving_api.py#L124" >}}

### `load_model` function

```python
keras.saving.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)
```

Loads a model saved via `model.save()`.

**Arguments**

- **filepath**: `str` or `pathlib.Path` object, path to the saved model file.
- **custom_objects**: Optional dictionary mapping names (strings) to custom classes or functions to be considered during deserialization.
- **compile**: Boolean, whether to compile the model after loading.
- **safe_mode**: Boolean, whether to disallow unsafe `lambda` deserialization. When `safe_mode=False`, loading an object has the potential to trigger arbitrary code execution. This argument is only applicable to the Keras v3 model format. Defaults to `True`.

**Returns**

A Keras model instance. If the original model was compiled, and the argument `compile=True` is set, then the returned model will be compiled. Otherwise, the model will be left uncompiled.

**Example**

```python
model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(3,)),
    keras.layers.Softmax()])
model.save("model.keras")
loaded_model = keras.saving.load_model("model.keras")
x = np.random.random((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))
```

Note that the model variables may have different name values (`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded. It is recommended that you use layer attributes to access specific variables, e.g. `model.get_layer("dense_1").kernel`.
