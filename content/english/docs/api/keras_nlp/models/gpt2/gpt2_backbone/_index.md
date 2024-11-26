---
title: GPT2Backbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/gpt2/gpt2_backbone.py#L17" >}}

### `GPT2Backbone` class

```python
keras_nlp.models.GPT2Backbone(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    dropout=0.1,
    max_sequence_length=1024,
    dtype=None,
    **kwargs
)
```

GPT-2 core network with hyperparameters.

This network implements a Transformer-based decoder network,
Generative Pretrained Transformer-2 (GPT-2), as described in
["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
It includes the embedding lookups and transformer layers.

The default constructor gives a fully customizable, randomly initialized
GPT-2 model with any number of layers, heads, and embedding
dimensions. To load preset architectures and weights, use the `from_preset`
constructor.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind. The underlying model is provided by a
third party and subject to a separate license, available
[here](https://github.com/openai/gpt-2).

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
  can consume. If `None`, `max_sequence_length` uses the value from
  sequence length. This determines the variable shape for positional
  embeddings.
- **dtype**: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
  for the models computations and weights. Note that some
  computations, such as softmax and layer normalization will always
  be done a float32 precision regardless of dtype.

**Example**

```python
input_data = {
    "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
}
# Pretrained GPT-2 decoder.
model = keras_hub.models.GPT2Backbone.from_preset("gpt2_base_en")
model(input_data)
# Randomly initialized GPT-2 decoder with custom config.
model = keras_hub.models.GPT2Backbone(
    vocabulary_size=50257,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    intermediate_dim=3072,
    max_sequence_length=1024,
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
GPT2Backbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name                | Parameters | Description                                                                                          |
| -------------------------- | ---------- | ---------------------------------------------------------------------------------------------------- |
| gpt2_base_en               | 124.44M    | 12-layer GPT-2 model where case is maintained. Trained on WebText.                                   |
| gpt2_medium_en             | 354.82M    | 24-layer GPT-2 model where case is maintained. Trained on WebText.                                   |
| gpt2_large_en              | 774.03M    | 36-layer GPT-2 model where case is maintained. Trained on WebText.                                   |
| gpt2_extra_large_en        | 1.56B      | 48-layer GPT-2 model where case is maintained. Trained on WebText.                                   |
| gpt2_base_en_cnn_dailymail | 124.44M    | 12-layer GPT-2 model where case is maintained. Finetuned on the CNN/DailyMail summarization dataset. |

### `token_embedding` property

```python
keras_nlp.models.GPT2Backbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.
