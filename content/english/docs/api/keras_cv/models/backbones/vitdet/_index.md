---
title: ViTDet backbones
toc: true
weight: 10
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/vit_det/vit_det_backbone.py#L34" >}}

### `ViTDetBackbone` class

```python
keras_cv.models.ViTDetBackbone(
    include_rescaling,
    input_shape=(1024, 1024, 3),
    input_tensor=None,
    patch_size=16,
    embed_dim=768,
    depth=12,
    mlp_dim=3072,
    num_heads=12,
    out_chans=256,
    use_bias=True,
    use_abs_pos=True,
    use_rel_pos=True,
    window_size=14,
    global_attention_indices=[2, 5, 8, 11],
    layer_norm_epsilon=1e-06,
    **kwargs
)
```

A ViT image encoder that uses a windowed transformer encoder and
relative positional encodings.

**Arguments**

- **input_shape (tuple[int], optional)**: The size of the input image in
  `(H, W, C)` format. Defaults to `(1024, 1024, 3)`.
- **input_tensor (KerasTensor, optional)**: Output of
  `keras.layers.Input()`) to use as image input for the model.
  Defaults to `None`.
- **include_rescaling (bool, optional)**: Whether to rescale the inputs. If
  set to `True`, inputs will be passed through a
  `Rescaling(1/255.0)` layer. Defaults to `False`.
- **patch_size (int, optional)**: the patch size to be supplied to the
  Patching layer to turn input images into a flattened sequence of
  patches. Defaults to `16`.
- **embed_dim (int, optional)**: The latent dimensionality to be projected
  into in the output of each stacked windowed transformer encoder.
  Defaults to `768`.
- **depth (int, optional)**: The number of transformer encoder layers to
  stack in the Vision Transformer. Defaults to `12`.
- **mlp_dim (int, optional)**: The dimensionality of the hidden Dense
  layer in the transformer MLP head. Defaults to `768*4`.
- **num_heads (int, optional)**: the number of heads to use in the
  `MultiHeadAttentionWithRelativePE` layer of each transformer
  encoder. Defaults to `12`.
- **out_chans (int, optional)**: The number of channels (features) in the
  output (image encodings). Defaults to `256`.
- **use_bias (bool, optional)**: Whether to use bias to project the keys,
  queries, and values in the attention layer. Defaults to `True`.
- **use_abs_pos (bool, optional)**: Whether to add absolute positional
  embeddings to the output patches. Defaults to `True`.
- **use_rel_pos (bool, optional)**: Whether to use relative positional
  emcodings in the attention layer. Defaults to `True`.
- **window_size (int, optional)**: The size of the window for windowed
  attention in the transformer encoder blocks. Defaults to `14`.
- **global_attention_indices (list, optional)**: Indexes for blocks using
  global attention. Defaults to `[2, 5, 8, 11]`.
- **layer_norm_epsilon (int, optional)**: The epsilon to use in the layer
  normalization blocks in transformer encoder. Defaults to `1e-6`.

**References**

- [Segment Anything paper](https://arxiv.org/abs/2304.02643)
  - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
  - [Detectron2](https://github.com/facebookresearch/detectron2)

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
ViTDetBackbone.from_preset()
```

Instantiate ViTDetBackbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "vitdet_base", "vitdet_large", "vitdet_huge", "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b".
  If looking for a preset with pretrained weights, choose one of
  "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.ViTDetBackbone.from_preset(
    "vitdet_base_sa1b",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.ViTDetBackbone.from_preset(
    "vitdet_base_sa1b",
    load_weights=False,
```

| Preset name       | Parameters | Description                                                                                                                                                            |
| ----------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| vitdet_base       | 89.67M     | Detectron2 ViT basebone with 12 transformer encoders with embed dim 768 and attention layers with 12 heads with global attention on encoders 2, 5, 8, and 11.          |
| vitdet_large      | 308.28M    | Detectron2 ViT basebone with 24 transformer encoders with embed dim 1024 and attention layers with 16 heads with global attention on encoders 5, 11, 17, and 23.       |
| vitdet_huge       | 637.03M    | Detectron2 ViT basebone model with 32 transformer encoders with embed dim 1280 and attention layers with 16 heads with global attention on encoders 7, 15, 23, and 31. |
| vitdet_base_sa1b  | 89.67M     | A base Detectron2 ViT backbone trained on the SA1B dataset.                                                                                                            |
| vitdet_large_sa1b | 308.28M    | A large Detectron2 ViT backbone trained on the SA1B dataset.                                                                                                           |
| vitdet_huge_sa1b  | 637.03M    | A huge Detectron2 ViT backbone trained on the SA1B dataset.                                                                                                            |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/vit_det/vit_det_aliases.py#L47" >}}

### `ViTDetBBackbone` class

```python
keras_cv.models.ViTDetBBackbone(
    include_rescaling,
    input_shape=(1024, 1024, 3),
    input_tensor=None,
    patch_size=16,
    embed_dim=768,
    depth=12,
    mlp_dim=3072,
    num_heads=12,
    out_chans=256,
    use_bias=True,
    use_abs_pos=True,
    use_rel_pos=True,
    window_size=14,
    global_attention_indices=[2, 5, 8, 11],
    layer_norm_epsilon=1e-06,
    **kwargs
)
```

VitDetBBackbone model.

**Reference**

- [Detectron2](https://github.com/facebookresearch/detectron2)
  - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
  - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Example**

```python
input_data = np.ones(shape=(1, 1024, 1024, 3))
# Randomly initialized backbone
model = VitDetBBackbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/vit_det/vit_det_aliases.py#L71" >}}

### `ViTDetLBackbone` class

```python
keras_cv.models.ViTDetLBackbone(
    include_rescaling,
    input_shape=(1024, 1024, 3),
    input_tensor=None,
    patch_size=16,
    embed_dim=768,
    depth=12,
    mlp_dim=3072,
    num_heads=12,
    out_chans=256,
    use_bias=True,
    use_abs_pos=True,
    use_rel_pos=True,
    window_size=14,
    global_attention_indices=[2, 5, 8, 11],
    layer_norm_epsilon=1e-06,
    **kwargs
)
```

VitDetLBackbone model.

**Reference**

- [Detectron2](https://github.com/facebookresearch/detectron2)
  - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
  - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Example**

```python
input_data = np.ones(shape=(1, 1024, 1024, 3))
# Randomly initialized backbone
model = VitDetLBackbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/vit_det/vit_det_aliases.py#L95" >}}

### `ViTDetHBackbone` class

```python
keras_cv.models.ViTDetHBackbone(
    include_rescaling,
    input_shape=(1024, 1024, 3),
    input_tensor=None,
    patch_size=16,
    embed_dim=768,
    depth=12,
    mlp_dim=3072,
    num_heads=12,
    out_chans=256,
    use_bias=True,
    use_abs_pos=True,
    use_rel_pos=True,
    window_size=14,
    global_attention_indices=[2, 5, 8, 11],
    layer_norm_epsilon=1e-06,
    **kwargs
)
```

VitDetHBackbone model.

**Reference**

- [Detectron2](https://github.com/facebookresearch/detectron2)
  - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
  - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning]({{< relref "/docs/guides/transfer_learning/" >}}).

**Example**

```python
input_data = np.ones(shape=(1, 1024, 1024, 3))
# Randomly initialized backbone
model = VitDetHBackbone()
output = model(input_data)
```
