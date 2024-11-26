---
title: VitDet model
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/vit_det/vit_det_backbone.py#L11" >}}

### `ViTDetBackbone` class

```python
keras_hub.models.ViTDetBackbone(
    hidden_size,
    num_layers,
    intermediate_dim,
    num_heads,
    global_attention_layer_indices,
    image_shape=(None, None, 3),
    patch_size=16,
    num_output_channels=256,
    use_bias=True,
    use_abs_pos=True,
    use_rel_pos=True,
    window_size=14,
    layer_norm_epsilon=1e-06,
    **kwargs
)
```

An implementation of ViT image encoder.

The ViTDetBackbone uses a windowed transformer encoder and relative
positional encodings. The code has been adapted from [Segment Anything
paper](https://arxiv.org/abs/2304.02643), [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything) and [Detectron2](https://github.com/facebookresearch/detectron2).

**Arguments**

- **hidden_size (int)**: The latent dimensionality to be projected
  into in the output of each stacked windowed transformer encoder.
- **num_layers (int)**: The number of transformer encoder layers to
  stack in the Vision Transformer.
- **intermediate_dim (int)**: The dimensionality of the hidden Dense
  layer in the transformer MLP head.
- **num_heads (int)**: the number of heads to use in the
  `MultiHeadAttentionWithRelativePE` layer of each transformer
  encoder.
- **global_attention_layer_indices (list)**: Indexes for blocks using
  global attention.
- **image_shape (tuple[int], optional)**: The size of the input image in
  `(H, W, C)` format. Defaults to `(None, None, 3)`.
- **patch_size (int, optional)**: the patch size to be supplied to the
  Patching layer to turn input images into a flattened sequence of
  patches. Defaults to `16`.
- **num_output_channels (int, optional)**: The number of channels (features)
  in the output (image encodings). Defaults to `256`.
- **use_bias (bool, optional)**: Whether to use bias to project the keys,
  queries, and values in the attention layer. Defaults to `True`.
- **use_abs_pos (bool, optional)**: Whether to add absolute positional
  embeddings to the output patches. Defaults to `True`.
- **use_rel_pos (bool, optional)**: Whether to use relative positional
  emcodings in the attention layer. Defaults to `True`.
- **window_size (int, optional)**: The size of the window for windowed
  attention in the transformer encoder blocks. Defaults to `14`.
- **layer_norm_epsilon (int, optional)**: The epsilon to use in the layer
  normalization blocks in transformer encoder. Defaults to `1e-6`.

**Examples**

```python
input_data = np.ones((2, 224, 224, 3), dtype="float32")
# Pretrained ViTDetBackbone backbone.
model = keras_hub.models.ViTDetBackbone.from_preset("vit_det")
model(input_data)
# Randomly initialized ViTDetBackbone backbone with a custom config.
model = keras_hub.models.ViTDetBackbone(
        image_shape = (16, 16, 3),
        patch_size = 2,
        hidden_size = 4,
        num_layers = 2,
        global_attention_layer_indices = [2, 5, 8, 11],
        intermediate_dim = 4 * 4,
        num_heads = 2,
        num_output_channels = 2,
        window_size = 2,
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
ViTDetBackbone.from_preset(preset, load_weights=True, **kwargs)
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
