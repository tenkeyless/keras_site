---
title: Model export for inference
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L460" >}}

### `export` method

```python
Model.export(filepath, format="tf_saved_model", verbose=True)
```

Create a TF SavedModel artifact for inference.

**Note:** This can currently only be used with the TensorFlow or JAX backends.

This method lets you export a model to a lightweight SavedModel artifact that contains the model's forward pass only (its `call()` method) and can be served via e.g. TF-Serving. The forward pass is registered under the name `serve()` (see example below).

The original code of the model (including any custom layers you may have used) is _no longer_ necessary to reload the artifact – it is entirely standalone.

**Arguments**

- **filepath**: `str` or `pathlib.Path` object. Path where to save the artifact.
- **verbose**: whether to print all the variables of the exported model.

**Example**

```python
# Create the artifact
model.export("path/to/location")

# Later, in a different process/environment...
reloaded_artifact = tf.saved_model.load("path/to/location")
predictions = reloaded_artifact.serve(input_data)
```

If you would like to customize your serving endpoints, you can use the lower-level [`keras.export.ExportArchive`]({{< relref "/docs/api/models/model_saving_apis/export#exportarchive-class" >}}) class. The `export()` method relies on `ExportArchive` internally.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/export/export_lib.py#L20" >}}

### `ExportArchive` class

```python
keras.export.ExportArchive()
```

ExportArchive is used to write SavedModel artifacts (e.g. for inference).

If you have a Keras model or layer that you want to export as SavedModel for serving (e.g. via TensorFlow-Serving), you can use `ExportArchive` to configure the different serving endpoints you need to make available, as well as their signatures. Simply instantiate an `ExportArchive`, use `track()` to register the layer(s) or model(s) to be used, then use the `add_endpoint()` method to register a new serving endpoint. When done, use the `write_out()` method to save the artifact.

The resulting artifact is a SavedModel and can be reloaded via [`tf.saved_model.load`](https://www.tensorflow.org/api_docs/python/tf/saved_model/load).

**Examples**

Here's how to export a model for inference.

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
export_archive.write_out("path/to/location")

# Elsewhere, we can reload the artifact and serve it.
# The endpoint we added is available as a method:
serving_model = tf.saved_model.load("path/to/location")
outputs = serving_model.serve(inputs)
```

Here's how to export a model with one endpoint for inference and one endpoint for a training-mode forward pass (e.g. with dropout on).

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="call_inference",
    fn=lambda x: model.call(x, training=False),
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
export_archive.add_endpoint(
    name="call_training",
    fn=lambda x: model.call(x, training=True),
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
export_archive.write_out("path/to/location")
```

**Note on resource tracking:**

`ExportArchive` is able to automatically track all [`tf.Variables`](https://www.tensorflow.org/api_docs/python/tf/Variables) used by its endpoints, so most of the time calling `.track(model)` is not strictly required. However, if your model uses lookup layers such as `IntegerLookup`, `StringLookup`, or `TextVectorization`, it will need to be tracked explicitly via `.track(model)`.

Explicit tracking is also required if you need to be able to access the properties `variables`, `trainable_variables`, or `non_trainable_variables` on the revived archive.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/export/export_lib.py#L200" >}}

### `add_endpoint` method

```python
ExportArchive.add_endpoint(name, fn, input_signature=None, jax2tf_kwargs=None)
```

Register a new serving endpoint.

**Arguments**

- **name**: Str, name of the endpoint.
- **fn**: A function. It should only leverage resources (e.g. [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) objects or [`tf.lookup.StaticHashTable`](https://www.tensorflow.org/api_docs/python/tf/lookup/StaticHashTable) objects) that are available on the models/layers tracked by the `ExportArchive` (you can call `.track(model)` to track a new model). The shape and dtype of the inputs to the function must be known. For that purpose, you can either 1) make sure that `fn` is a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) that has been called at least once, or 2) provide an `input_signature` argument that specifies the shape and dtype of the inputs (see below).
- **input_signature**: Used to specify the shape and dtype of the inputs to `fn`. List of [`tf.TensorSpec`](https://www.tensorflow.org/api_docs/python/tf/TensorSpec) objects (one per positional input argument of `fn`). Nested arguments are allowed (see below for an example showing a Functional model with 2 input arguments).
- **jax2tf_kwargs**: Optional. A dict for arguments to pass to `jax2tf`. Supported only when the backend is JAX. See documentation for [`jax2tf.convert`](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md). The values for `native_serialization` and `polymorphic_shapes`, if not provided, are automatically computed.

**Returns**

The [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) wrapping `fn` that was added to the archive.

**Example**

Adding an endpoint using the `input_signature` argument when the model has a single input argument:

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
```

Adding an endpoint using the `input_signature` argument when the model has two positional input arguments:

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    ],
)
```

Adding an endpoint using the `input_signature` argument when the model has one input argument that is a list of 2 tensors (e.g. a Functional model with 2 inputs):

```python
model = keras.Model(inputs=[x1, x2], outputs=outputs)

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[
        [
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        ],
    ],
)
```

This also works with dictionary inputs:

```python
model = keras.Model(inputs={"x1": x1, "x2": x2}, outputs=outputs)

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[
        {
            "x1": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            "x2": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        },
    ],
)
```

Adding an endpoint that is a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function):

```python
@tf.function()
def serving_fn(x):
    return model(x)

# The function must be traced, i.e. it must be called at least once.
serving_fn(tf.random.normal(shape=(2, 3)))

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(name="serve", fn=serving_fn)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/export/export_lib.py#L420" >}}

### `add_variable_collection` method

```python
ExportArchive.add_variable_collection(name, variables)
```

Register a set of variables to be retrieved after reloading.

**Arguments**

- **name**: The string name for the collection.
- **variables**: A tuple/list/set of [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) instances.

**Example**

```python
export_archive = ExportArchive()
export_archive.track(model)
# Register an endpoint
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
# Save a variable collection
export_archive.add_variable_collection(
    name="optimizer_variables", variables=model.optimizer.variables)
export_archive.write_out("path/to/location")

# Reload the object
revived_object = tf.saved_model.load("path/to/location")
# Retrieve the variables
optimizer_variables = revived_object.optimizer_variables
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/export/export_lib.py#L119" >}}

### `track` method

```python
ExportArchive.track(resource)
```

Track the variables (and other assets) of a layer or model.

By default, all variables used by an endpoint function are automatically tracked when you call `add_endpoint()`. However, non-variables assets such as lookup tables need to be tracked manually. Note that lookup tables used by built-in Keras layers (`TextVectorization`, `IntegerLookup`, `StringLookup`) are automatically tracked in `add_endpoint()`.

**Arguments**

- **resource**: A trackable TensorFlow resource.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/export/export_lib.py#L468" >}}

### `write_out` method

```python
ExportArchive.write_out(filepath, options=None, verbose=True)
```

Write the corresponding SavedModel to disk.

**Arguments**

- **filepath**: `str` or `pathlib.Path` object. Path where to save the artifact.
- **options**: [`tf.saved_model.SaveOptions`](https://www.tensorflow.org/api_docs/python/tf/saved_model/SaveOptions) object that specifies SavedModel saving options.
- **verbose**: whether to print all the variables of an exported SavedModel.

**Note on TF-Serving**: all endpoints registered via `add_endpoint()` are made visible for TF-Serving in the SavedModel artifact. In addition, the first endpoint registered is made visible under the alias `"serving_default"` (unless an endpoint with the name `"serving_default"` was already registered manually), since TF-Serving requires this endpoint to be set.
