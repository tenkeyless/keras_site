---
title: DistilBertTextClassifier model
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/distil_bert/distil_bert_text_classifier.py#L16" >}}

### `DistilBertTextClassifier` class

```python
keras_nlp.models.DistilBertTextClassifier(
    backbone,
    num_classes,
    preprocessor=None,
    activation=None,
    hidden_dim=None,
    dropout=0.2,
    **kwargs
)
```

An end-to-end DistilBERT model for classification tasks.

This model attaches a classification head to a
`keras_hub.model.DistilBertBackbone` instance, mapping from the backbone
outputs to logits suitable for a classification task. For usage of
this model with pre-trained weights, see the `from_preset()` constructor.

This model can optionally be configured with a `preprocessor` layer, in
which case it will automatically apply preprocessing to raw inputs during
`fit()`, `predict()`, and `evaluate()`. This is done by default when
creating the model with `from_preset()`.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind. The underlying model is provided by a
third party and subject to a separate license, available
[here](https://github.com/huggingface/transformers).

**Arguments**

- **backbone**: A `keras_hub.models.DistilBert` instance.
- **num_classes**: int. Number of classes to predict.
- **preprocessor**: A [`keras_hub.models.DistilBertTextClassifierPreprocessor`]({{< relref "/docs/api/keras_hub/models/distil_bert/distil_bert_text_classifier_preprocessor#distilberttextclassifierpreprocessor-class" >}}) or `None`. If
  `None`, this model will not apply preprocessing, and inputs should
  be preprocessed before calling the model.
- **activation**: Optional `str` or callable. The
  activation function to use on the model outputs. Set
  `activation="softmax"` to return output probabilities.
  Defaults to `None`.
- **hidden_dim**: int. The size of the pooler layer.
- **dropout**: float. The dropout probability value, applied after the first
  dense layer.

**Examples**

Raw string data.

```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 3]
# Use a shorter sequence length.
preprocessor = keras_hub.models.DistilBertTextClassifierPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=128,
)
# Pretrained classifier.
classifier = keras_hub.models.DistilBertTextClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=4,
    preprocessor=preprocessor,
)
classifier.fit(x=features, y=labels, batch_size=2)
# Re-compile (e.g., with a new learning rate)
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
)
# Access backbone programmatically (e.g., to change `trainable`).
classifier.backbone.trainable = False
# Fit again.
classifier.fit(x=features, y=labels, batch_size=2)
```

Preprocessed integer data.

```python
features = {
    "token_ids": np.ones(shape=(2, 12), dtype="int32"),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2)
}
labels = [0, 3]
# Pretrained classifier without preprocessing.
classifier = keras_hub.models.DistilBertTextClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=4,
    preprocessor=None,
)
classifier.fit(x=features, y=labels, batch_size=2)
```

Custom backbone and vocabulary.

```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 3]
vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
vocab += ["The", "quick", "brown", "fox", "jumped", "."]
tokenizer = keras_hub.models.DistilBertTokenizer(
    vocabulary=vocab,
)
preprocessor = keras_hub.models.DistilBertTextClassifierPreprocessor(
    tokenizer=tokenizer,
    sequence_length=128,
)
backbone = keras_hub.models.DistilBertBackbone(
    vocabulary_size=30552,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    intermediate_dim=512,
    max_sequence_length=128,
)
classifier = keras_hub.models.DistilBertTextClassifier(
    backbone=backbone,
    preprocessor=preprocessor,
    num_classes=4,
)
classifier.fit(x=features, y=labels, batch_size=2)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
DistilBertTextClassifier.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name                 | Parameters | Description                                                                                                                         |
| --------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| distil_bert_base_en_uncased | 66.36M     | 6-layer DistilBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus using BERT as the teacher model. |
| distil_bert_base_en         | 65.19M     | 6-layer DistilBERT model where case is maintained. Trained on English Wikipedia + BooksCorpus using BERT as the teacher model.      |
| distil_bert_base_multi      | 134.73M    | 6-layer DistilBERT model where case is maintained. Trained on Wikipedias of 104 languages                                           |

### `backbone` property

```python
keras_nlp.models.DistilBertTextClassifier.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_nlp.models.DistilBertTextClassifier.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.
