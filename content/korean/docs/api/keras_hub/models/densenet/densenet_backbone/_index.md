---
title: DensNetBackbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/densenet/densenet_backbone.py#L9" >}}

### `DenseNetBackbone` class

```python
keras_hub.models.DenseNetBackbone(
    stackwise_num_repeats,
    image_shape=(None, None, 3),
    compression_ratio=0.5,
    growth_rate=32,
    **kwargs
)
```

Instantiates the DenseNet architecture.

This class implements a DenseNet backbone as described in
[Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993).

**Arguments**

- **stackwise_num_repeats**: list of ints, number of repeated convolutional
  blocks per dense block.
- **image_shape**: optional shape tuple, defaults to (None, None, 3).
- **compression_ratio**: float, compression rate at transition layers,
  defaults to 0.5.
- **growth_rate**: int, number of filters added by each dense block,
  defaults to 32

**Examples**

```python
input_data = np.ones(shape=(8, 224, 224, 3))
# Pretrained backbone
model = keras_hub.models.DenseNetBackbone.from_preset("densenet121_imagenet")
model(input_data)
# Randomly initialized backbone with a custom config
model = keras_hub.models.DenseNetBackbone(
    stackwise_num_repeats=[6, 12, 24, 16],
)
model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
DenseNetBackbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name           | Parameters | Description                                                                              |
| --------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| densenet_121_imagenet | 7.04M      | 121-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution. |
| densenet_169_imagenet | 12.64M     | 169-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution. |
| densenet_201_imagenet | 18.32M     | 201-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution. |
