---
title: MiTBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/mit/mit_backbone.py#L11" >}}

### `MiTBackbone` class

```python
keras_hub.models.MiTBackbone(
    depths,
    num_layers,
    blockwise_num_heads,
    blockwise_sr_ratios,
    max_drop_path_rate,
    patch_sizes,
    strides,
    image_shape=(None, None, 3),
    hidden_dims=None,
    **kwargs
)
```

A backbone with feature pyramid outputs.

`FeaturePyramidBackbone` extends `Backbone` with a single `pyramid_outputs`
property for accessing the feature pyramid outputs of the model. Subclassers
should set the `pyramid_outputs` property during the model constructor.

**Example**

```python
input_data = np.random.uniform(0, 256, size=(2, 224, 224, 3))
# Convert to feature pyramid output format using ResNet.
backbone = ResNetBackbone.from_preset("resnet50")
model = keras.Model(
    inputs=backbone.inputs, outputs=backbone.pyramid_outputs
)
model(input_data)  # A dict containing the keys ["P2", "P3", "P4", "P5"]
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
MiTBackbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name            | Parameters | Description                                            |
| ---------------------- | ---------- | ------------------------------------------------------ |
| mit_b0_ade20k_512      | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks.  |
| mit_b1_ade20k_512      | 13.16M     | MiT (MixTransformer) model with 8 transformer blocks.  |
| mit_b2_ade20k_512      | 24.20M     | MiT (MixTransformer) model with 16 transformer blocks. |
| mit_b3_ade20k_512      | 44.08M     | MiT (MixTransformer) model with 28 transformer blocks. |
| mit_b4_ade20k_512      | 60.85M     | MiT (MixTransformer) model with 41 transformer blocks. |
| mit_b5_ade20k_640      | 81.45M     | MiT (MixTransformer) model with 52 transformer blocks. |
| mit_b0_cityscapes_1024 | 3.32M      | MiT (MixTransformer) model with 8 transformer blocks.  |
| mit_b1_cityscapes_1024 | 13.16M     | MiT (MixTransformer) model with 8 transformer blocks.  |
| mit_b2_cityscapes_1024 | 24.20M     | MiT (MixTransformer) model with 16 transformer blocks. |
| mit_b3_cityscapes_1024 | 44.08M     | MiT (MixTransformer) model with 28 transformer blocks. |
| mit_b4_cityscapes_1024 | 60.85M     | MiT (MixTransformer) model with 41 transformer blocks. |
| mit_b5_cityscapes_1024 | 81.45M     | MiT (MixTransformer) model with 52 transformer blocks. |
