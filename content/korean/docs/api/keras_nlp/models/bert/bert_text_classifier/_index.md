---
title: BertTextClassifier model
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/bert/bert_text_classifier.py#L12" >}}

### `BertTextClassifier` class

```python
keras_nlp.models.BertTextClassifier(
    backbone, num_classes, preprocessor=None, activation=None, dropout=0.1, **kwargs
)
```

An end-to-end BERT model for classification tasks.

This model attaches a classification head to a
`keras_hub.model.BertBackbone` instance, mapping from the backbone outputs
to logits suitable for a classification task. For usage of this model with
pre-trained weights, use the `from_preset()` constructor.

This model can optionally be configured with a `preprocessor` layer, in
which case it will automatically apply preprocessing to raw inputs during
`fit()`, `predict()`, and `evaluate()`. This is done by default when
creating the model with `from_preset()`.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind.

**Arguments**

- **backbone**: A [`keras_hub.models.BertBackbone`]({{< relref "/docs/api/keras_hub/models/bert/bert_backbone#bertbackbone-class" >}}) instance.
- **num_classes**: int. Number of classes to predict.
- **preprocessor**: A [`keras_hub.models.BertTextClassifierPreprocessor`]({{< relref "/docs/api/keras_hub/models/bert/bert_text_classifier_preprocessor#berttextclassifierpreprocessor-class" >}}) or `None`. If
  `None`, this model will not apply preprocessing, and inputs should
  be preprocessed before calling the model.
- **activation**: Optional `str` or callable. The
  activation function to use on the model outputs. Set
  `activation="softmax"` to return output probabilities.
  Defaults to `None`.
- **dropout**: float. The dropout probability value, applied after the dense
  layer.

**Examples**

Raw string data.

```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 3]
# Pretrained classifier.
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=4,
)
classifier.fit(x=features, y=labels, batch_size=2)
classifier.predict(x=features, batch_size=2)
# Re-compile (e.g., with a new learning rate).
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
    "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
}
labels = [0, 3]
# Pretrained classifier without preprocessing.
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased",
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
tokenizer = keras_hub.models.BertTokenizer(
    vocabulary=vocab,
)
preprocessor = keras_hub.models.BertTextClassifierPreprocessor(
    tokenizer=tokenizer,
    sequence_length=128,
)
backbone = keras_hub.models.BertBackbone(
    vocabulary_size=30552,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    intermediate_dim=512,
    max_sequence_length=128,
)
classifier = keras_hub.models.BertTextClassifier(
    backbone=backbone,
    preprocessor=preprocessor,
    num_classes=4,
)
classifier.fit(x=features, y=labels, batch_size=2)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
BertTextClassifier.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name               | Parameters | Description                                                                                     |
| ------------------------- | ---------- | ----------------------------------------------------------------------------------------------- |
| bert_tiny_en_uncased      | 4.39M      | 2-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.   |
| bert_small_en_uncased     | 28.76M     | 4-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.   |
| bert_medium_en_uncased    | 41.37M     | 8-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.   |
| bert_base_en_uncased      | 109.48M    | 12-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.  |
| bert_base_en              | 108.31M    | 12-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus.       |
| bert_base_zh              | 102.27M    | 12-layer BERT model. Trained on Chinese Wikipedia.                                              |
| bert_base_multi           | 177.85M    | 12-layer BERT model where case is maintained. Trained on trained on Wikipedias of 104 languages |
| bert_large_en_uncased     | 335.14M    | 24-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.  |
| bert_large_en             | 333.58M    | 24-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus.       |
| bert_tiny_en_uncased_sst2 | 4.39M      | The bert_tiny_en_uncased backbone model fine-tuned on the SST-2 sentiment analysis dataset.     |

### `backbone` property

```python
keras_nlp.models.BertTextClassifier.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_nlp.models.BertTextClassifier.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.
