---
title: Weights-only saving & loading
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L313" >}}

### `save_weights` method

```python
Model.save_weights(filepath, overwrite=True)
```

Saves all layer weights to a `.weights.h5` file.

**Arguments**

- **filepath**: `str` or `pathlib.Path` object. Path where to save the model. Must end in `.weights.h5`.
- **overwrite**: Whether we should overwrite any existing model at the target location, or instead ask the user via an interactive prompt.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/models/model.py#L326" >}}

### `load_weights` method

```python
Model.load_weights(filepath, skip_mismatch=False, **kwargs)
```

Load weights from a file saved via `save_weights()`.

Weights are loaded based on the network's topology. This means the architecture should be the same as when the weights were saved. Note that layers that don't have weights are not taken into account in the topological ordering, so adding or removing layers is fine as long as they don't have weights.

**Partial weight loading**

If you have modified your model, for instance by adding a new layer (with weights) or by changing the shape of the weights of a layer, you can choose to ignore errors and continue loading by setting `skip_mismatch=True`. In this case any layer with mismatching weights will be skipped. A warning will be displayed for each skipped layer.

**Arguments**

- **filepath**: String, path to the weights file to load. It can either be a `.weights.h5` file or a legacy `.h5` weights file.
- **skip_mismatch**: Boolean, whether to skip loading of layers where there is a mismatch in the number of weights, or a mismatch in the shape of the weights.
