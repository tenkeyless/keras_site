---
title: ImageSegmenter
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/image_segmenter.py#L7" >}}

### `ImageSegmenter` class

```python
keras_hub.models.ImageSegmenter(*args, compile=True, **kwargs)
```

Base class for all image segmentation tasks.

`ImageSegmenter` tasks wrap a [`keras_hub.models.Task`]({{< relref "/docs/api/keras_hub/base_classes/task#task-class" >}}) and
a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) to create a model that can be used for
image segmentation.

All `ImageSegmenter` tasks include a `from_preset()` constructor which can
be used to load a pre-trained config and weights.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
ImageSegmenter.from_preset(preset, load_weights=True, **kwargs)
```

Instantiate a [`keras_hub.models.Task`]({{< relref "/docs/api/keras_hub/base_classes/task#task-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Task` subclass, you can run `cls.presets.keys()` to list all
built-in presets available on the class.

This constructor can be called in one of two ways. Either from a task
specific base class like `keras_hub.models.CausalLM.from_preset()`, or
from a model class like `keras_hub.models.BertTextClassifier.from_preset()`.
If calling from the a base class, the subclass of the returning object
will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, saved weights will be loaded into
  the model architecture. If `False`, all weights will be
  randomly initialized.

**Examples**

```python
# Load a Gemma generative task.
causal_lm = keras_hub.models.CausalLM.from_preset(
    "gemma_2b_en",
)
# Load a Bert classification task.
model = keras_hub.models.TextClassifier.from_preset(
    "bert_base_en",
    num_classes=2,
)
```

| Preset name                        | Parameters | Description                                                                                                                                                                                     |
| ---------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| deeplab_v3_plus_resnet50_pascalvoc | 39.19M     | DeepLabV3+ model with ResNet50 as image encoder and trained on augmented Pascal VOC dataset by Semantic Boundaries Dataset(SBD)which is having categorical accuracy of 90.01 and 0.63 Mean IoU. |
| sam_base_sa1b                      | 93.74M     | The base SAM model trained on the SA1B dataset.                                                                                                                                                 |
| sam_large_sa1b                     | 641.09M    | The large SAM model trained on the SA1B dataset.                                                                                                                                                |
| sam_huge_sa1b                      | 312.34M    | The huge SAM model trained on the SA1B dataset.                                                                                                                                                 |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/image_segmenter.py#L19" >}}

### `compile` method

```python
ImageSegmenter.compile(optimizer="auto", loss="auto", metrics="auto", **kwargs)
```

Configures the `ImageSegmenter` task for training.

The `ImageSegmenter` task extends the default compilation signature of
[`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) with defaults for `optimizer`, `loss`, and
`metrics`. To override these defaults, pass any value
to these arguments during compilation.

**Arguments**

- **optimizer**: `"auto"`, an optimizer name, or a `keras.Optimizer`
  instance. Defaults to `"auto"`, which uses the default optimizer
  for the given model and task. See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and
  `keras.optimizers` for more info on possible `optimizer` values.
- **loss**: `"auto"`, a loss name, or a [`keras.losses.Loss`]({{< relref "/docs/api/losses#loss-class" >}}) instance.
  Defaults to `"auto"`, where a
  [`keras.losses.SparseCategoricalCrossentropy`]({{< relref "/docs/api/losses/probabilistic_losses#sparsecategoricalcrossentropy-class" >}}) loss will be
  applied for the classification task. See
  [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and `keras.losses` for more info on
  possible `loss` values.
- **metrics**: `"auto"`, or a list of metrics to be evaluated by
  the model during training and testing. Defaults to `"auto"`,
  where a [`keras.metrics.SparseCategoricalAccuracy`]({{< relref "/docs/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class" >}}) will be
  applied to track the accuracy of the model during training.
  See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and `keras.metrics` for
  more info on possible `metrics` values.
- **\*\*kwargs**: See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) for a full list of arguments
  supported by the compile method.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L238" >}}

### `save_to_preset` method

```python
ImageSegmenter.save_to_preset(preset_dir)
```

Save task to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

### `preprocessor` property

```python
keras_hub.models.ImageSegmenter.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.

### `backbone` property

```python
keras_hub.models.ImageSegmenter.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.
