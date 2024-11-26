---
title: MaskedLM
toc: true
weight: 10
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/masked_lm.py#L7" >}}

### `MaskedLM` class

```python
keras_nlp.models.MaskedLM()
```

Base class for masked language modeling tasks.

`MaskedLM` tasks wrap a [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) and
a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) to create a model that can be used for
unsupervised fine-tuning with a masked language modeling loss.

When calling `fit()`, all input will be tokenized, and random tokens in
the input sequence will be masked. These positions of these masked tokens
will be fed as an additional model input, and the original value of the
tokens predicted by the model outputs.

All `MaskedLM` tasks include a `from_preset()` constructor which can be used
to load a pre-trained config and weights.

**Example**

```python
# Load a Bert MaskedLM with pre-trained weights.
masked_lm = keras_hub.models.MaskedLM.from_preset(
    "bert_base_en",
)
masked_lm.fit(train_ds)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
MaskedLM.from_preset(preset, load_weights=True, **kwargs)
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

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/masked_lm.py#L36" >}}

### `compile` method

```python
MaskedLM.compile(optimizer="auto", loss="auto", weighted_metrics="auto", **kwargs)
```

Configures the `MaskedLM` task for training.

The `MaskedLM` task extends the default compilation signature of
[`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) with defaults for `optimizer`, `loss`, and
`weighted_metrics`. To override these defaults, pass any value
to these arguments during compilation.

Note that because training inputs include padded tokens which are
excluded from the loss, it is almost always a good idea to compile with
`weighted_metrics` and not `metrics`.

**Arguments**

- **optimizer**: `"auto"`, an optimizer name, or a `keras.Optimizer`
  instance. Defaults to `"auto"`, which uses the default optimizer
  for the given model and task. See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and
  `keras.optimizers` for more info on possible `optimizer` values.
- **loss**: `"auto"`, a loss name, or a [`keras.losses.Loss`]({{< relref "/docs/api/losses#loss-class" >}}) instance.
  Defaults to `"auto"`, where a
  [`keras.losses.SparseCategoricalCrossentropy`]({{< relref "/docs/api/losses/probabilistic_losses#sparsecategoricalcrossentropy-class" >}}) loss will be
  applied for the token classification `MaskedLM` task. See
  [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and `keras.losses` for more info on
  possible `loss` values.
- **weighted_metrics**: `"auto"`, or a list of metrics to be evaluated by
  the model during training and testing. Defaults to `"auto"`,
  where a [`keras.metrics.SparseCategoricalAccuracy`]({{< relref "/docs/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class" >}}) will be
  applied to track the accuracy of the model at guessing masked
  token values. See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and `keras.metrics` for
  more info on possible `weighted_metrics` values.
- **\*\*kwargs**: See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) for a full list of arguments
  supported by the compile method.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L238" >}}

### `save_to_preset` method

```python
MaskedLM.save_to_preset(preset_dir)
```

Save task to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

### `preprocessor` property

```python
keras_nlp.models.MaskedLM.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.

### `backbone` property

```python
keras_nlp.models.MaskedLM.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.
