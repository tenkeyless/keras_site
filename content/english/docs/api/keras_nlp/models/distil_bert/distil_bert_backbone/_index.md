---
title: DistilBertBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/distil_bert/distil_bert_backbone.py#L15" >}}

### `DistilBertBackbone` class

```python
keras_nlp.models.DistilBertBackbone(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    dropout=0.1,
    max_sequence_length=512,
    dtype=None,
    **kwargs
)
```

A DistilBERT encoder network.

This network implements a bi-directional Transformer-based encoder as
described in ["DistilBERT, a distilled version of BERT: smaller, faster,
cheaper and lighter"](https://arxiv.org/abs/1910.01108). It includes the
embedding lookups and transformer layers, but not the masked language model
or classification task networks.

The default constructor gives a fully customizable, randomly initialized
DistilBERT encoder with any number of layers, heads, and embedding
dimensions. To load preset architectures and weights, use the
`from_preset()` constructor.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind. The underlying model is provided by a
third party and subject to a separate license, available
[here](https://github.com/huggingface/transformers).

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
- **dtype**: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
  for model computations and weights. Note that some computations,
  such as softmax and layer normalization, will always be done at
  float32 precision regardless of dtype.

**Examples**

```python
input_data = {
    "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
}
# Pretrained DistilBERT encoder.
model = keras_hub.models.DistilBertBackbone.from_preset(
    "distil_bert_base_en_uncased"
)
model(input_data)
# Randomly initialized DistilBERT encoder with custom config.
model = keras_hub.models.DistilBertBackbone(
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
DistilBertBackbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name                 | Parameters | Description                                                                                                                         |
| --------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| distil_bert_base_en_uncased | 66.36M     | 6-layer DistilBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus using BERT as the teacher model. |
| distil_bert_base_en         | 65.19M     | 6-layer DistilBERT model where case is maintained. Trained on English Wikipedia + BooksCorpus using BERT as the teacher model.      |
| distil_bert_base_multi      | 134.73M    | 6-layer DistilBERT model where case is maintained. Trained on Wikipedias of 104 languages                                           |

### `token_embedding` property

```python
keras_nlp.models.DistilBertBackbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.
