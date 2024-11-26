---
title: StableDiffusion3Backbone model
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/stable_diffusion_3/stable_diffusion_3_backbone.py#L184" >}}

### `StableDiffusion3Backbone` class

```python
keras_hub.models.StableDiffusion3Backbone(
    mmdit_patch_size,
    mmdit_hidden_dim,
    mmdit_num_layers,
    mmdit_num_heads,
    mmdit_position_size,
    vae,
    clip_l,
    clip_g,
    t5=None,
    latent_channels=16,
    output_channels=3,
    num_train_timesteps=1000,
    shift=3.0,
    height=None,
    width=None,
    data_format=None,
    dtype=None,
    **kwargs
)
```

Stable Diffusion 3 core network with hyperparameters.

This backbone imports CLIP and T5 models as text encoders and implements the
base MMDiT and VAE networks for the Stable Diffusion 3 model.

The default constructor gives a fully customizable, randomly initialized
MMDiT and VAE models with any hyperparameters. To load preset architectures
and weights, use the `from_preset` constructor.

**Arguments**

- **mmdit_patch_size**: int. The size of each square patch in the input image
  in MMDiT.
- **mmdit_hidden_dim**: int. The size of the transformer hidden state at the
  end of each transformer layer in MMDiT.
- **mmdit_num_layers**: int. The number of transformer layers in MMDiT.
- **mmdit_num_heads**: int. The number of attention heads for each
  transformer in MMDiT.
- **mmdit_position_size**: int. The size of the height and width for the
  position embedding in MMDiT.
- **vae**: The VAE used for transformations between pixel space and latent
  space.
- **clip_l**: The CLIP text encoder for encoding the inputs.
- **clip_g**: The CLIP text encoder for encoding the inputs.
- **t5**: optional The T5 text encoder for encoding the inputs.
- **latent_channels**: int. The number of channels in the latent. Defaults to
  `16`.
- **output_channels**: int. The number of channels in the output. Defaults to
  `3`.
- **num_train_timesteps**: int. The number of diffusion steps to train the
  model. Defaults to `1000`.
- **shift**: float. The shift value for the timestep schedule. Defaults to
  `3.0`.
- **height**: optional int. The output height of the image.
- **width**: optional int. The output width of the image.
- **data_format**: `None` or str. If specified, either `"channels_last"` or
  `"channels_first"`. The ordering of the dimensions in the
  inputs. `"channels_last"` corresponds to inputs with shape
  `(batch_size, height, width, channels)`
  while `"channels_first"` corresponds to inputs with shape
  `(batch_size, channels, height, width)`. It defaults to the
  `image_data_format` value found in your Keras config file at
  `~/.keras/keras.json`. If you never set it, then it will be
  `"channels_last"`.
- **dtype**: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
  for the models computations and weights. Note that some
  computations, such as softmax and layer normalization will always
  be done a float32 precision regardless of dtype.

**Example**

```python
# Pretrained Stable Diffusion 3 model.
model = keras_hub.models.StableDiffusion3Backbone.from_preset(
    "stable_diffusion_3_medium"
)
# Randomly initialized Stable Diffusion 3 model with custom config.
vae = keras_hub.models.VAEBackbone(...)
clip_l = keras_hub.models.CLIPTextEncoder(...)
clip_g = keras_hub.models.CLIPTextEncoder(...)
model = keras_hub.models.StableDiffusion3Backbone(
    mmdit_patch_size=2,
    mmdit_num_heads=4,
    mmdit_hidden_dim=256,
    mmdit_depth=4,
    mmdit_position_size=192,
    vae=vae,
    clip_l=clip_l,
    clip_g=clip_g,
)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

```python
StableDiffusion3Backbone.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name               | Parameters | Description                                                                                                                             |
| ------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| stable_diffusion_3_medium | 2.99B      | 3 billion parameter, including CLIP L and CLIP G text encoders, MMDiT generative model, and VAE autoencoder. Developed by Stability AI. |
