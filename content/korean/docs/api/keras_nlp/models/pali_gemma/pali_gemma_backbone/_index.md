---
title: PaliGemmaBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/pali_gemma/pali_gemma_backbone.py#L16" >}}

### `PaliGemmaBackbone` class

```python
keras_nlp.models.PaliGemmaBackbone(
    vocabulary_size,
    image_size,
    num_layers,
    num_query_heads,
    num_key_value_heads,
    hidden_dim,
    intermediate_dim,
    head_dim,
    vit_patch_size,
    vit_num_heads,
    vit_hidden_dim,
    vit_num_layers,
    vit_intermediate_dim=None,
    vit_pooling=None,
    vit_classifier_activation=None,
    vit_name=None,
    layer_norm_epsilon=1e-06,
    dropout=0,
    dtype=None,
    **kwargs
)
```

PaliGemma core network with hyperparameters.

This backbone implements the mixed-modality PaliGemma architecture. It
contains a Visual Transformer network, as well as text token embedding
layer, followed by a backend-agnostic concatenation operation to
construct a sequence of representations of mixed type embeddings (visual
and textual). Then, the concatenated sequence is passed through a series
of Mixed Modality Decoder Blocks. The returned value from calling this model
represents probabilistic values for output tokens.

For a higher-level object for text-generation,
see [`keras_hub.models.PaliGemmaCausalLM`]({{< relref "/docs/api/keras_hub/models/pali_gemma/pali_gemma_causal_lm#paligemmacausallm-class" >}}).

The default constructor gives a fully customizable, randomly initialized
PaliGemma model with any number of vit layers, heads, embedding
dimensions, and equivalent configuration for Paligemma Decoder layers. To
load preset architectures and weights, use the `from_preset` constructor.

**Arguments**

- **vocabulary_size**: int. The size of the token vocabulary.
- **image_size**: int. The resolution of the image in both width and height.
  Note: input images must be square.
- **num_layers**: int. The number of transformer mixed decoder layers.
- **num_query_heads**: int. The number of heads for the query projections in
  the mixed decoder attention layer.
- **num_key_value_heads**: int. The number of heads for the key and value
  projections in the mixed decoder attention layers.
- **hidden_dim**: int. The size of the transformer hidden state at the end
  of each mixed transformer layer.
- **intermediate_dim**: int. The output dimension of the first Dense layer in
  a two-layer feedforward network for each transformer decoder block.
- **head_dim**: int. The size of each attention head in the mixed decoder.
- **vit_patch_size**: int. The size of each square patch in the input image.
- **vit_num_heads**: int. The number of attention heads for the vision(image)
  transformer encoder.
- **vit_hidden_dim**: int. The size of the transformer hidden state at the end
  of each vision transformer layer.
- **vit_num_layers**: int. The number of vision transformer layers.
- **vit_intermediate_dim**: int. The output dimension of the first Dense layer
  in a two-layer feedforward network for vision transformer.
- **vit_pooling**: string. The encoded vision embeddings are pooled using the
  specified polling setting. The accepted values are `"map"`, `"gap"`,
  `"0"` or `"none"`. Defaults to `"none"`.
- **vit_classifier_activation**: activation function. The activation that
  is used for final output classification in the vision transformer.
- **vit_name**: string. The name used for vision transformer layers.
- **layer_norm_epsilon**: float. The epsilon value user for every layer norm
  in all transformer blocks.
- **dropout**: float. Dropout probability for the Transformer decoder blocks.
- **dtype**: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
  for the models computations and weights. Note that some
  computations, such as softmax and layer normalization will always
  be done a float32 precision regardless of dtype.

**Example**

```python
input_data = {
    "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    "images": np.random.uniform(size=(1, 224, 224, 3)),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
}
# Pretrained PaliGemma decoder.
model = keras_hub.models.PaliGemmaBackbone.from_preset("pali_gemma_mix_224")
model(input_data)
# Randomly initialized PaliGemma decoder with custom config.
model = keras_hub.models.PaliGemmaBackbone(
    vocabulary_size=50257,
    images_size=224,
    num_layers=12,
    num_query_heads=12,
    num_key_value_heads=1,
    hidden_dim=768,
    intermediate_dim=3072,
    head_dim=64,
    vit_patch_size=14,
    vit_num_heads=8,
    vit_hidden_dim=768,
    vit_intermediate_dim=3072,
    vit_num_layers=2,
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
PaliGemmaBackbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name           | Parameters | Description                                                 |
| --------------------- | ---------- | ----------------------------------------------------------- |
| pali_gemma_3b_mix_224 | 2.92B      | image size 224, mix fine tuned, text sequence length is 256 |
| pali_gemma_3b_mix_448 | 2.92B      | image size 448, mix fine tuned, text sequence length is 512 |
| pali_gemma_3b_224     | 2.92B      | image size 224, pre trained, text sequence length is 128    |
| pali_gemma_3b_448     | 2.92B      | image size 448, pre trained, text sequence length is 512    |
| pali_gemma_3b_896     | 2.93B      | image size 896, pre trained, text sequence length is 512    |

### `token_embedding` property

```python
keras_nlp.models.PaliGemmaBackbone.token_embedding
```

A [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.
