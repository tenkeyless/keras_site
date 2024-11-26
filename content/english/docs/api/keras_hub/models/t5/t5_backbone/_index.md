---
title: T5Backbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/t5/t5_backbone.py#L12" >}}

### `T5Backbone` class

```python
keras_hub.models.T5Backbone(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    key_value_dim=None,
    dropout=0.1,
    activation="relu",
    use_gated_activation=True,
    layer_norm_epsilon=1e-06,
    tie_embedding_weights=True,
    dtype=None,
    **kwargs
)
```

T5 encoder-decoder backbone model.

T5 is a LLM pretrained on a mix of unsupervised and supervised tasks,
where each task is converted to a sequence-to-sequence format.
T5 works well on a variety of tasks out-of-the-box by prepending
various prefixex to the input sequence, e.g., for translation:
`"translate English to German: ..."`, for summarization:
`"summarize: ..."`.

T5 was introduced in
[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

The default constructor gives a fully customizable, randomly initialized T5
model with any number of layers, heads, and embedding dimensions. To load
preset architectures and weights, use the `from_preset` constructor.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind.

**Arguments**

- **vocabulary_size**: int. The size of the token vocabulary.
- **num_layers**: int. The number of Transformer layers.
- **num_heads**: int. The number of attention heads for each Transformer.
  The hidden size must be divisible by the number of attention heads.
- **hidden_dim**: int. The hidden size of the Transformer layers.
- **intermediate_dim**: int. The output dimension of the first Dense layer in
  a two-layer feedforward network for each Transformer layer.
- **key_value_dim**: int. The dimension of each head of the key/value
  projections in the multi-head attention layers. Defaults to
  hidden_dim / num_heads.
- **dropout**: float. Dropout probability for the Transformer layers.
- **activation**: activation function (or activation string name). The
  activation to be used in the inner dense blocks of the
  Transformer layers. Defaults to `"relu"`.
- **use_gated_activation**: boolean. Whether to use activation gating in
  the inner dense blocks of the Transformer layers.
  The original T5 architecture didn't use gating, but more
  recent versions do. Defaults to `True`.
- **layer_norm_epsilon**: float. Epsilon factor to be used in the
  layer normalization layers in the Transformer layers.
- **tie_embedding_weights**: boolean. If `True`, the weights of the token
  embedding and the weights projecting language model outputs from
  `hidden_dim`.
- **dtype**: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
  for model computations and weights. Note that some computations,
  such as softmax and layer normalization, will always be done at
  float32 precision regardless of dtype.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
T5Backbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name      | Parameters | Description                                                           |
| ---------------- | ---------- | --------------------------------------------------------------------- |
| t5_small_multi   | 0          | 8-layer T5 model. Trained on the Colossal Clean Crawled Corpus (C4).  |
| t5_base_multi    | 0          | 12-layer T5 model. Trained on the Colossal Clean Crawled Corpus (C4). |
| t5_large_multi   | 0          | 24-layer T5 model. Trained on the Colossal Clean Crawled Corpus (C4). |
| flan_small_multi | 0          | 8-layer T5 model. Trained on the Colossal Clean Crawled Corpus (C4).  |
| flan_base_multi  | 0          | 12-layer T5 model. Trained on the Colossal Clean Crawled Corpus (C4). |
| flan_large_multi | 0          | 24-layer T5 model. Trained on the Colossal Clean Crawled Corpus (C4). |

### `token_embedding` property

```python
keras_hub.models.T5Backbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.
