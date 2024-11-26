---
title: Keras weights file editor
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L36" >}}

### `KerasFileEditor` class

```python
keras.saving.KerasFileEditor(filepath)
```

Utility to inspect, edit, and resave Keras weights files.

You will find this class useful when adapting an old saved weights file after having made architecture changes to a model.

**Arguments**

- **filepath**: The path to a local file to inspect and edit.

**Examples**

```python
editor = KerasFileEditor("my_model.weights.h5")

# Displays current contents
editor.summary()

# Remove the weights of an existing layer
editor.delete_object("layers/dense_2")

# Add the weights of a new layer
editor.add_object("layers/einsum_dense", weights={"0": ..., "1": ...})

# Save the weights of the edited model
editor.resave_weights("edited_model.weights.h5")
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L110" >}}

### `summary` method

```python
KerasFileEditor.summary()
```

Prints the weight structure of the opened file.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L114" >}}

### `compare` method

```python
KerasFileEditor.compare(reference_model)
```

Compares the opened file to a reference model.

This method will list all mismatches between the currently opened file and the provided reference model.

**Arguments**

- **reference_model**: Model instance to compare to.

**Returns**

- **Dict with the following keys**: `'status'`, `'error_count'`, `'match_count'`. Status can be `'success'` or `'error'`. `'error_count'` is the number of mismatches found. `'match_count'` is the number of matching weights found.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L414" >}}

### `save` method

```python
KerasFileEditor.save(filepath)
```

Save the edited weights file.

**Arguments**

- **filepath**: Path to save the file to. Must be a `.weights.h5` file.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L304" >}}

### `rename_object` method

```python
KerasFileEditor.rename_object(object_name, new_name)
```

Rename an object in the file (e.g. a layer).

**Arguments**

- **object_name**: String, name or path of the object to rename (e.g. `"dense_2"` or `"layers/dense_2"`).
- **new_name**: String, new name of the object.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L320" >}}

### `delete_object` method

```python
KerasFileEditor.delete_object(object_name)
```

Removes an object from the file (e.g. a layer).

**Arguments**

- **object_name**: String, name or path of the object to delete (e.g. `"dense_2"` or `"layers/dense_2"`).

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L334" >}}

### `add_object` method

```python
KerasFileEditor.add_object(object_path, weights)
```

Add a new object to the file (e.g. a layer).

**Arguments**

- **object_path**: String, full path of the object to add (e.g. `"layers/dense_2"`).
- **weights**: Dict mapping weight names to weight values (arrays), e.g. `{"0": kernel_value, "1": bias_value}`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L367" >}}

### `delete_weight` method

```python
KerasFileEditor.delete_weight(object_name, weight_name)
```

Removes a weight from an existing object.

**Arguments**

- **object_name**: String, name or path of the object from which to remove the weight (e.g. `"dense_2"` or `"layers/dense_2"`).
- **weight_name**: String, name of the weight to delete (e.g. `"0"`).

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/saving/file_editor.py#L390" >}}

### `add_weights` method

```python
KerasFileEditor.add_weights(object_name, weights)
```

Add one or more new weights to an existing object.

**Arguments**

- **object_name**: String, name or path of the object to add the weights to (e.g. `"dense_2"` or `"layers/dense_2"`).
- **weights**: Dict mapping weight names to weight values (arrays), e.g. `{"0": kernel_value, "1": bias_value}`.
