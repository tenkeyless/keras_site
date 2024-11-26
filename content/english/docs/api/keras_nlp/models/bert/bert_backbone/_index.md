---
title: BertBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/bert/bert_backbone.py#L17" >}}

### `BertBackbone` class

```python
keras_nlp.models.BertBackbone(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    dropout=0.1,
    max_sequence_length=512,
    num_segments=2,
    dtype=None,
    **kwargs
)
```

A BERT encoder network.

This class implements a bi-directional Transformer-based encoder as
described in ["BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
embedding lookups and transformer layers, but not the masked language model
or next sentence prediction heads.

The default constructor gives a fully customizable, randomly initialized
BERT encoder with any number of layers, heads, and embedding dimensions. To
load preset architectures and weights, use the `from_preset()` constructor.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind.

**Arguments**

- **vocabulary_size**: int. The size of the token vocabulary.
- **num_layers**: int. The number of transformer layers.
- **num_heads**: int. The number of attention heads for each transformer.
  The hidden size must be divisible by the number of attention heads.
- **hidden_dim**: int. The size of the transformer encoding and pooler layers.
- **intermediate_dim**: int. The output dimension of the first Dense layer in
  a two-layer feedforward network for each transformer.
- **dropout**: float. Dropout probability for the Transformer encoder.
- **max_sequence_length**: int. The maximum sequence length that this encoder
  can consume. If None, `max_sequence_length` uses the value from
  sequence length. This determines the variable shape for positional
  embeddings.
- **num_segments**: int. The number of types that the 'segment_ids' input can
  take.
- **dtype**: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
  for model computations and weights. Note that some computations,
  such as softmax and layer normalization, will always be done at
  float32 precision regardless of dtype.

**Examples**

```python
input_data = {
    "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
}
# Pretrained BERT encoder.
model = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
model(input_data)
# Randomly initialized BERT encoder with a custom config.
model = keras_hub.models.BertBackbone(
    vocabulary_size=30552,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    intermediate_dim=512,
    max_sequence_length=128,
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
BertBackbone.from_preset(preset, load_weights=True, **kwargs)
```

Instantiate a [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as a
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

This constructor can be called in one of two ways. Either from the base
class like `keras_hub.models.Backbone.from_preset()`, or from
a model class like `keras_hub.models.GemmaBackbone.from_preset()`.
If calling from the base class, the subclass of the returning object
will be inferred from the config in the preset directory.

For any `Backbone` subclass, you can run `cls.presets.keys()` to list
all built-in presets available on the class.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the
  model architecture. If `False`, the weights will be randomly
  initialized.

**Examples**

```python
# Load a Gemma backbone with pre-trained weights.
model = keras_hub.models.Backbone.from_preset(
    "gemma_2b_en",
)
# Load a Bert backbone with a pre-trained config and random weights.
model = keras_hub.models.Backbone.from_preset(
    "bert_base_en",
    load_weights=False,
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

### `token_embedding` property

```python
keras_nlp.models.BertBackbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.
