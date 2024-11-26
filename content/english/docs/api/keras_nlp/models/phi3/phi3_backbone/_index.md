---
title: Phi3Backbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/phi3/phi3_backbone.py#L16" >}}

### `Phi3Backbone` class

```python
keras_nlp.models.Phi3Backbone(
    vocabulary_size,
    num_layers,
    hidden_dim,
    intermediate_dim,
    num_query_heads,
    num_key_value_heads,
    layer_norm_epsilon=1e-06,
    dropout=0.0,
    max_sequence_length=4096,
    pretraining_sequence_length=4096,
    rope_max_wavelength=10000,
    rope_scaling_type=None,
    rope_scaling_short_factor=None,
    rope_scaling_long_factor=None,
    dtype=None,
    **kwargs
)
```

Phi-3 core network with hyperparameters.

This network implements a Transformer-based decoder network,
Phi-3, as described in ["Phi-3 Technical Report"](https://arxiv.org/pdf/2404.14219.pdf).
It includes the embedding lookups and transformer layers.

The default constructor gives a fully customizable, randomly initialized
phi-3 model with any number of layers, heads, and embedding
dimensions. To load preset architectures and weights, use the `from_preset`
constructor.

**Arguments**

- **vocabulary_size (int)**: The size of the token vocabulary.
- **num_layers (int)**: The number of transformer layers.
- **hidden_dim (int)**: The size of the embeddings and the hidden states of
  the transformer layers.
- **intermediate_dim (int)**: The output dimension of the first Dense layer in
  a three-layer feedforward network for each transformer.
- **num_query_heads (int)**: The number of query attention heads for each
  transformer layer.
- **num_key_value_heads (int)**: The number of key and value attention heads
  for each transformer layer.
- **layer_norm_epsilon (float, optional)**: Epsilon for the RMS layernorm
  layers in the transformer decoder. Defaults to `1e-6`.
- **dropout**: (float, optional): Dropout probability for the Transformer
  decoder.
- **max_sequence_length (int, optional)**: The maximum sequence length
  that this model might ever be used with. Defaults to `4096`.
- **pretraining_sequence_length (int, optional)**: The maximum sequence length
  that the model was pretrained with. Defaults to `4096`.
- **rope_max_wavelength (int, optional)**: The maximum angular wavelength of
  the sine/cosine curves, for rotary embeddings. Defaults to `10000`.
- **rope_scaling_type (str, optional)**: The type of the rope scaling. Can be
  either `None` or `"su"`. `None` is for no rope scaling, `"su"` is
  for SuScaled rope, `"su"` is used when `max_sequence_length` is
  larger than `original_max_sequence_length`. Defaults to `None`.
- **rope_scaling_short_factor List[float]**: List of factors used to adjust
  rope frequencies when the `rope_scaling_type` is `"su"`. List must
  be of length `hidden_dim//num_query_heads//2`. It is used when
  `sequence_length` is smaller than `original_max_sequence_length`.
  Defaults to `None`.
- **rope_scaling_long_factor List[float]**: List of factors used to adjust
  rope frequencies when the `rope_scaling_type` is `"su"`. List must
  be of length `hidden_dim//num_query_heads//2`. It is used when
  `sequence_length` is larger than `original_max_sequence_length`.
  Defaults to `None`.
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
# Pretrained Phi3 decoder.
model = keras_hub.models.Phi3Backbone.from_preset(
    "phi3_mini_4k_instruct_en"
)
model(input_data)
# Randomly initialized Phi3 decoder with custom config.
model = keras_hub.models.Phi3Backbone(
    vocabulary_size=10,
    num_layers=2,
    hidden_dim=512,
    intermediate_dim=1024,
    num_query_heads=32,
    num_key_value_heads=8,
    layer_norm_epsilon=1e-6,
    dtype="float32"
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
Phi3Backbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name                | Parameters | Description                                                                                                                                                                                                                                                                   |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| phi3_mini_4k_instruct_en   | 3.82B      | 3.8 billion parameters, 32 layers, 4k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties.   |
| phi3_mini_128k_instruct_en | 3.82B      | 3.8 billion parameters, 32 layers, 128k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties. |

### `token_embedding` property

```python
keras_nlp.models.Phi3Backbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.
