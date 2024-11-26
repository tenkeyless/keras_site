---
title: Inpaint
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/inpaint.py#L18" >}}

### `Inpaint` class

```python
keras_hub.models.Inpaint()
```

Base class for image-to-image tasks.

`Inpaint` tasks wrap a [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) and
a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) to create a model that can be used for
generation and generative fine-tuning.

`Inpaint` tasks provide an additional, high-level `generate()` function
which can be used to generate image by token with a (image, mask, string)
in, image out signature.

All `Inpaint` tasks include a `from_preset()` constructor which can be
used to load a pre-trained config and weights.

**Example**

```python
# Load a Stable Diffusion 3 backbone with pre-trained weights.
reference_image = np.ones((1024, 1024, 3), dtype="float32")
reference_mask = np.ones((1024, 1024), dtype="float32")
inpaint = keras_hub.models.Inpaint.from_preset(
    "stable_diffusion_3_medium",
)
inpaint.generate(
    reference_image,
    reference_mask,
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
)
# Load a Stable Diffusion 3 backbone at bfloat16 precision.
inpaint = keras_hub.models.Inpaint.from_preset(
    "stable_diffusion_3_medium",
    dtype="bfloat16",
)
inpaint.generate(
    reference_image,
    reference_mask,
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
Inpaint.from_preset(preset, load_weights=True, **kwargs)
```

Instantiate a [`keras_hub.models.Task`]({{< relref "/docs/api/keras_hub/base_classes/task#task-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Task` subclass, you can run `cls.presets.keys()` to list all
built-in presets available on the class.

This constructor can be called in one of two ways. Either from a task
specific base class like `keras_hub.models.CausalLM.from_preset()`, or
from a model class like `keras_hub.models.BertTextClassifier.from_preset()`.
If calling from the a base class, the subclass of the returning object
will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, saved weights will be loaded into
  the model architecture. If `False`, all weights will be
  randomly initialized.

**Examples**

```python
# Load a Gemma generative task.
causal_lm = keras_hub.models.CausalLM.from_preset(
    "gemma_2b_en",
)
# Load a Bert classification task.
model = keras_hub.models.TextClassifier.from_preset(
    "bert_base_en",
    num_classes=2,
)
```

| Preset name               | Parameters | Description                                                                                                                             |
| ------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| stable_diffusion_3_medium | 2.99B      | 3 billion parameter, including CLIP L and CLIP G text encoders, MMDiT generative model, and VAE autoencoder. Developed by Stability AI. |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/inpaint.py#L79" >}}

### `compile` method

```python
Inpaint.compile(optimizer="auto", loss="auto", metrics="auto", **kwargs)
```

Configures the `Inpaint` task for training.

The `Inpaint` task extends the default compilation signature of
[`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) with defaults for `optimizer`, `loss`, and
`metrics`. To override these defaults, pass any value
to these arguments during compilation.

**Arguments**

- **optimizer**: `"auto"`, an optimizer name, or a `keras.Optimizer`
  instance. Defaults to `"auto"`, which uses the default optimizer
  for the given model and task. See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and
  `keras.optimizers` for more info on possible `optimizer` values.
- **loss**: `"auto"`, a loss name, or a [`keras.losses.Loss`]({{< relref "/docs/api/losses#loss-class" >}}) instance.
  Defaults to `"auto"`, where a
  [`keras.losses.MeanSquaredError`]({{< relref "/docs/api/losses/regression_losses#meansquarederror-class" >}}) loss will be applied. See
  [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and `keras.losses` for more info on
  possible `loss` values.
- **metrics**: `"auto"`, or a list of metrics to be evaluated by
  the model during training and testing. Defaults to `"auto"`,
  where a [`keras.metrics.MeanSquaredError`]({{< relref "/docs/api/metrics/regression_metrics#meansquarederror-class" >}}) will be applied to
  track the loss of the model during training. See
  [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) and `keras.metrics` for more info on
  possible `metrics` values.
- **\*\*kwargs**: See [`keras.Model.compile`]({{< relref "/docs/api/models/model_training_apis#compile-method" >}}) for a full list of arguments
  supported by the compile method.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L238" >}}

### `save_to_preset` method

```python
Inpaint.save_to_preset(preset_dir)
```

Save task to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

### `preprocessor` property

```python
keras_hub.models.Inpaint.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.

### `backbone` property

```python
keras_hub.models.Inpaint.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/inpaint.py#L375" >}}

### `generate` method

```python
Inpaint.generate(inputs, num_steps, guidance_scale, strength, seed=None)
```

Generate image based on the provided `inputs`.

Typically, `inputs` is a dict with `"images"` `"masks"` and `"prompts"`
keys. `"images"` are reference images within a value range of
`[-1.0, 1.0]`, which will be resized to `self.backbone.height` and
`self.backbone.width`, then encoded into latent space by the VAE
encoder. `"masks"` are mask images with a boolean dtype, where white
pixels are repainted while black pixels are preserved. `"prompts"` are
strings that will be tokenized and encoded by the text encoder.

Some models support a `"negative_prompts"` key, which helps steer the
model away from generating certain styles and elements. To enable this,
add `"negative_prompts"` to the input dict.

If `inputs` are a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), outputs will be generated
"batch-by-batch" and concatenated. Otherwise, all inputs will be
processed as batches.

**Arguments**

- **inputs**: python data, tensor data, or a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). The format
  must be one of the following:
  - A dict with `"images"`, `"masks"`, `"prompts"` and/or
    `"negative_prompts"` keys.
  - A [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) with `"images"`, `"masks"`, `"prompts"`
    and/or `"negative_prompts"` keys.
- **num_steps**: int. The number of diffusion steps to take.
- **guidance_scale**: float. The classifier free guidance scale defined in
  [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). A higher scale encourages
  generating images more closely related to the prompts, typically
  at the cost of lower image quality.
- **strength**: float. Indicates the extent to which the reference
  `images` are transformed. Must be between `0.0` and `1.0`. When
  `strength=1.0`, `images` is essentially ignore and added noise
  is maximum and the denoising process runs for the full number of
  iterations specified in `num_steps`.
- **seed**: optional int. Used as a random seed.
