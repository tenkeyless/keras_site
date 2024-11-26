---
title: FalconBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/falcon/falcon_backbone.py#L13" >}}

### `FalconBackbone` class

```python
keras_hub.models.FalconBackbone(
    vocabulary_size,
    num_layers,
    num_attention_heads,
    hidden_dim,
    intermediate_dim,
    layer_norm_epsilon=1e-05,
    attention_dropout_rate=0,
    feedforward_dropout_rate=0,
    dtype=None,
    **kwargs
)
```

The Falcon core architecure.

This network implements a Transformer-based decoder-only network,
[Falcon](https://arxiv.org/abs/2306.01116).

**Arguments**

- **vocabulary_size**: int. The size of the token vocabulary.
- **num_layers**: int. The number of transformer layers.
- **num_attention_heads**: int. The number of attention heads for each transformer.
  The hidden size must be divisible by the number of attention heads.
- **hidden_dim**: int. The dimensionality of the embeddings and hidden states.
- **intermediate_dim**: int. The output dimension of the first Dense layer in
  the MLP network of each transformer.
- **layer_norm_epsilon**: float. Epsilon for the layer normalization layers in
  the transformer decoder.
- **attention_dropout_rate**: float. Dropout probability for the attention.
- **feedforward_dropout_rate**: flaot. Dropout probability for the feedforward.
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
# Pretrained Falcon decoder.
# TODO: Update the preset.
model = keras_hub.models.FalconBackbone.from_preset("falcon_preset")
model(input_data)
# Randomly initialized Falcon decoder with a custom config.
model = keras_hub.models.FalconBackbone(
    vocabulary_size=10,
    num_layers=2,
    num_attention_heads=2,
    hidden_dim=32,
    intermediate_dim=32*4,
    layer_norm_epsilon=1e-5,
    attention_dropout_rate=0,
    feedforward_dropout_rate=0,
    dtype="float32",
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
FalconBackbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name             | Parameters | Description                                                                                      |
| ----------------------- | ---------- | ------------------------------------------------------------------------------------------------ |
| falcon_refinedweb_1b_en | 1.31B      | 24-layer Falcon model (Falcon with 1B parameters), trained on 350B tokens of RefinedWeb dataset. |

### `token_embedding` property

```python
keras_hub.models.FalconBackbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.
