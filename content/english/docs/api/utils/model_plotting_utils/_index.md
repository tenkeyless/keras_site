---
title: Model plotting utilities
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/model_visualization.py#L372" >}}

### `plot_model` function

```python
keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=False,
    show_trainable=False,
    **kwargs
)
```

Converts a Keras model to dot format and save to a file.

**Example**

```python
inputs = ...
outputs = ...
model = keras.Model(inputs=inputs, outputs=outputs)
dot_img_file = '/tmp/model_1.png'
keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
```

**Arguments**

- **model**: A Keras model instance
- **to_file**: File name of the plot image.
- **show_shapes**: whether to display shape information.
- **show_dtype**: whether to display layer dtypes.
- **show_layer_names**: whether to display layer names.
- **rankdir**: `rankdir` argument passed to PyDot,
  a string specifying the format of the plot: `"TB"`
  creates a vertical plot; `"LR"` creates a horizontal plot.
- **expand_nested**: whether to expand nested Functional models
  into clusters.
- **dpi**: Image resolution in dots per inch.
- **show_layer_activations**: Display layer activations (only for layers that
  have an `activation` property).
- **show_trainable**: whether to display if a layer is trainable.

**Returns**

A Jupyter notebook Image object if Jupyter is installed.
This enables in-line display of the model plots in notebooks.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/model_visualization.py#L201" >}}

### `model_to_dot` function

```python
keras.utils.model_to_dot(
    model,
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    subgraph=False,
    show_layer_activations=False,
    show_trainable=False,
    **kwargs
)
```

Convert a Keras model to dot format.

**Arguments**

- **model**: A Keras model instance.
- **show_shapes**: whether to display shape information.
- **show_dtype**: whether to display layer dtypes.
- **show_layer_names**: whether to display layer names.
- **rankdir**: `rankdir` argument passed to PyDot,
  a string specifying the format of the plot: `"TB"`
  creates a vertical plot; `"LR"` creates a horizontal plot.
- **expand_nested**: whether to expand nested Functional models
  into clusters.
- **dpi**: Image resolution in dots per inch.
- **subgraph**: whether to return a `pydot.Cluster` instance.
- **show_layer_activations**: Display layer activations (only for layers that
  have an `activation` property).
- **show_trainable**: whether to display if a layer is trainable.

**Returns**

A `pydot.Dot` instance representing the Keras model or
a `pydot.Cluster` instance representing nested model if
`subgraph=True`.
