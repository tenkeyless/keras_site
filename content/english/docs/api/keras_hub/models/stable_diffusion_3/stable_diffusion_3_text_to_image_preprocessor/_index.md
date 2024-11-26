---
title: StableDiffusion3TextToImagePreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/stable_diffusion_3/stable_diffusion_3_text_to_image_preprocessor.py#L11" >}}

### `StableDiffusion3TextToImagePreprocessor` class

```python
keras_hub.models.StableDiffusion3TextToImagePreprocessor(
    clip_l_preprocessor, clip_g_preprocessor, t5_preprocessor=None, **kwargs
)
```

Stable Diffusion 3 text-to-image model preprocessor.

This preprocessing layer is meant for use with
[`keras_hub.models.StableDiffusion3TextToImage`]({{< relref "/docs/api/keras_hub/models/stable_diffusion_3/stable_diffusion_3_text_to_image#stablediffusion3texttoimage-class" >}}).

For use with generation, the layer exposes one methods
`generate_preprocess()`.

**Arguments**

- **clip_l_preprocessor**: A `keras_hub.models.CLIPPreprocessor` instance.
- **clip_g_preprocessor**: A `keras_hub.models.CLIPPreprocessor` instance.
- **t5_preprocessor**: A optional [`keras_hub.models.T5Preprocessor`]({{< relref "/docs/api/keras_hub/models/t5/t5_preprocessor#t5preprocessor-class" >}}) instance.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
StableDiffusion3TextToImagePreprocessor.from_preset(
    preset, config_file="preprocessor.json", **kwargs
)
```

Instantiate a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Preprocessor` subclass, you can run `cls.presets.keys()` to
list all built-in presets available on the class.

As there are usually multiple preprocessing classes for a given model,
this method should be called on a specific subclass like
`keras_hub.models.BertTextClassifierPreprocessor.from_preset()`.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.

**Examples**

```python
# Load a preprocessor for Gemma generation.
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en",
)
# Load a preprocessor for Bert classification.
preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en",
)
```

| Preset name               | Parameters | Description                                                                                                                             |
| ------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| stable_diffusion_3_medium | 2.99B      | 3 billion parameter, including CLIP L and CLIP G text encoders, MMDiT generative model, and VAE autoencoder. Developed by Stability AI. |
