---
title: MistralBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/mistral/mistral_backbone.py#L21" >}}

### `MistralBackbone` class

```python
keras_hub.models.MistralBackbone(
    vocabulary_size,
    num_layers,
    num_query_heads,
    hidden_dim,
    intermediate_dim,
    num_key_value_heads,
    rope_max_wavelength=10000,
    rope_scaling_factor=1.0,
    layer_norm_epsilon=1e-06,
    sliding_window=512,
    dropout=0,
    dtype=None,
    **kwargs
)
```

The Mistral Transformer core architecture with hyperparameters.

This network implements a Transformer-based decoder network,
Mistral, as described in
["Mistral 7B"](https://arxiv.org/pdf/2310.06825.pdf).
It includes the embedding lookups and transformer layers.

The default constructor gives a fully customizable, randomly initialized
Mistral model with any number of layers, heads, and embedding
dimensions. To load preset architectures and weights, use the `from_preset`
constructor.

**Arguments**

- **vocabulary_size (int)**: The size of the token vocabulary.
- **num_layers (int)**: The number of transformer layers.
- **num_query_heads (int)**: The number of query attention heads for
  each transformer.
- **hidden_dim (int)**: The size of the transformer encoding and pooling layers.
- **intermediate_dim (int)**: The output dimension of the first Dense layer in a
  three-layer feedforward network for each transformer.
- **num_key_value_heads (int)**: The number of key and value attention heads for
  each transformer.
- **rope_max_wavelength (int, optional)**: The maximum angular wavelength of the
  sine/cosine curves, for rotary embeddings. Defaults to `10000`.
- **rope_scaling_factor (float, optional)**: The scaling factor for calculation
  of roatary embedding. Defaults to `1.0`.
- **layer_norm_epsilon (float, optional)**: Epsilon for the layer normalization
  layers in the transformer decoder. Defaults to `1e-6`.
- **sliding_window (int, optional)**: The sliding window for the mistral
  attention layers. This controls the maximum cache size for the attention
  layers in each transformer decoder. Only `sliding_window` number of tokens
  are saved in the cache and used to generate the next token.
  Defaults to `512`.
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
# Pretrained Mistral decoder.
model = keras_hub.models.MistralBackbone.from_preset("mistral7b_base_en")
model(input_data)
# Randomly initialized Mistral decoder with custom config.
model = keras_hub.models.MistralBackbone(
    vocabulary_size=10,
    hidden_dim=512,
    num_layers=2,
    num_query_heads=32,
    num_key_value_heads=8,
    intermediate_dim=1024,
    sliding_window=512,
    layer_norm_epsilon=1e-6,
    dtype="float32"
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
MistralBackbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name                | Parameters | Description                           |
| -------------------------- | ---------- | ------------------------------------- |
| mistral_7b_en              | 7.24B      | Mistral 7B base model                 |
| mistral_instruct_7b_en     | 7.24B      | Mistral 7B instruct model             |
| mistral_0.2_instruct_7b_en | 7.24B      | Mistral 7B instruct Version 0.2 model |

### `token_embedding` property

```python
keras_hub.models.MistralBackbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L194" >}}

### `enable_lora` method

```python
MistralBackbone.enable_lora(rank)
```

Enable Lora on the backbone.

Calling this method will freeze all weights on the backbone,
while enabling Lora on the query & value `EinsumDense` layers
of the attention layers.
