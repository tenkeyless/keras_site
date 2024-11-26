---
title: RobertaMaskedLM model
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/roberta/roberta_masked_lm.py#L15" >}}

### `RobertaMaskedLM` class

```python
keras_hub.models.RobertaMaskedLM(backbone, preprocessor=None, **kwargs)
```

An end-to-end RoBERTa model for the masked language modeling task.

This model will train RoBERTa on a masked language modeling task.
The model will predict labels for a number of masked tokens in the
input data. For usage of this model with pre-trained weights, see the
`from_preset()` method.

This model can optionally be configured with a `preprocessor` layer, in
which case inputs can be raw string features during `fit()`, `predict()`,
and `evaluate()`. Inputs will be tokenized and dynamically masked during
training and evaluation. This is done by default when creating the model
with `from_preset()`.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind. The underlying model is provided by a
third party and subject to a separate license, available
[here](https://github.com/facebookresearch/fairseq).

**Arguments**

- **backbone**: A [`keras_hub.models.RobertaBackbone`]({{< relref "/docs/api/keras_hub/models/roberta/roberta_backbone#robertabackbone-class" >}}) instance.
- **preprocessor**: A [`keras_hub.models.RobertaMaskedLMPreprocessor`]({{< relref "/docs/api/keras_hub/models/roberta/roberta_masked_lm_preprocessor#robertamaskedlmpreprocessor-class" >}}) or
  `None`. If `None`, this model will not apply preprocessing, and
  inputs should be preprocessed before calling the model.

**Examples**

Raw string data.

```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
# Pretrained language model.
masked_lm = keras_hub.models.RobertaMaskedLM.from_preset(
    "roberta_base_en",
)
masked_lm.fit(x=features, batch_size=2)
# Re-compile (e.g., with a new learning rate).
masked_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
)
# Access backbone programmatically (e.g., to change `trainable`).
masked_lm.backbone.trainable = False
# Fit again.
masked_lm.fit(x=features, batch_size=2)
```

Preprocessed integer data.

```python
# Create a preprocessed dataset where 0 is the mask token.
features = {
    "token_ids": np.array([[1, 2, 0, 4, 0, 6, 7, 8]] * 2),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2),
    "mask_positions": np.array([[2, 4]] * 2)
}
# Labels are the original masked values.
labels = [[3, 5]] * 2
masked_lm = keras_hub.models.RobertaMaskedLM.from_preset(
    "roberta_base_en",
    preprocessor=None,
)
masked_lm.fit(x=features, y=labels, batch_size=2)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
RobertaMaskedLM.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name             | Parameters | Description                                                                                                             |
| ----------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| roberta_base_en         | 124.05M    | 12-layer RoBERTa model where case is maintained.Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText. |
| roberta_large_en        | 354.31M    | 24-layer RoBERTa model where case is maintained.Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText. |
| xlm_roberta_base_multi  | 277.45M    | 12-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages.                           |
| xlm_roberta_large_multi | 558.84M    | 24-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages.                           |

### `backbone` property

```python
keras_hub.models.RobertaMaskedLM.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_hub.models.RobertaMaskedLM.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.
