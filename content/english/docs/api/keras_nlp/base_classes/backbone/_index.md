---
title: Backbone
toc: true
weight: 1
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L11" >}}

### `Backbone` class

`keras_nlp.models.Backbone(*args, dtype=None, **kwargs)`

Base class for all `Backbone` models.

A `Backbone` is the basic architecture for a given NLP model. Unlike a [`keras_hub.models.Task`](/api/keras_hub/base_classes/task#task-class), a `Backbone` is not tailored to any specific loss function and training setup. A `Backbone` generally outputs the last hidden states of an architecture before any output predictions.

A `Backbone` can be used in one of two ways:

1.  Through a `Task` class, which will wrap and extend a `Backbone` so it can be used with high level Keras functions like `fit()`, `predict()` or `evaluate()`. `Task` classes are built with a particular training objective in mind (e.g. classification or language modeling).
2.  Directly, by extending underlying functional model with additional outputs and training setup. This is the most flexible approach, and can allow for any outputs, loss, or custom training loop.

All backbones include a `from_preset()` constructor which can be used to load a pre-trained config and weights.

**Example**

`# Load a BERT backbone with pre-trained weights. backbone = keras_hub.models.Backbone.from_preset(     "bert_base_en", ) # Load a GPT2 backbone with pre-trained weights at bfloat16 precision. backbone = keras_hub.models.Backbone.from_preset(     "gpt2_base_en",     dtype="bfloat16",     trainable=False, )`

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L127" >}}

### `from_preset` method

`Backbone.from_preset(preset, load_weights=True, **kwargs)`

Instantiate a [`keras_hub.models.Backbone`](/api/keras_hub/base_classes/backbone#backbone-class) from a model preset.

A preset is a directory of configs, weights and other file assets used to save and load a pre-trained model. The `preset` can be passed as a one of:

1.  a built-in preset identifier like `'bert_base_en'`
2.  a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3.  a Hugging Face handle like `'hf://user/bert_base_en'`
4.  a path to a local preset directory like `'./bert_base_en'`

This constructor can be called in one of two ways. Either from the base class like `keras_hub.models.Backbone.from_preset()`, or from a model class like `keras_hub.models.GemmaBackbone.from_preset()`. If calling from the base class, the subclass of the returning object will be inferred from the config in the preset directory.

For any `Backbone` subclass, you can run `cls.presets.keys()` to list all built-in presets available on the class.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the model architecture. If `False`, the weights will be randomly initialized.

**Examples**

`# Load a Gemma backbone with pre-trained weights. model = keras_hub.models.Backbone.from_preset(     "gemma_2b_en", )  # Load a Bert backbone with a pre-trained config and random weights. model = keras_hub.models.Backbone.from_preset(     "bert_base_en",     load_weights=False, )`

---

### `token_embedding` property

`keras_nlp.models.Backbone.token_embedding`

A [`keras.layers.Embedding`](/api/layers/core_layers/embedding#embedding-class) instance for embedding token ids.

This layer embeds integer token ids to the hidden dim of the model.

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L194" >}}

### `enable_lora` method

`Backbone.enable_lora(rank)`

Enable Lora on the backbone.

Calling this method will freeze all weights on the backbone, while enabling Lora on the query & value `EinsumDense` layers of the attention layers.

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L217" >}}

### `save_lora_weights` method

`Backbone.save_lora_weights(filepath)`

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L246" >}}

### `load_lora_weights` method

`Backbone.load_lora_weights(filepath)`

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/backbone.py#L185" >}}

### `save_to_preset` method

`Backbone.save_to_preset(preset_dir)`

Save backbone to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

---
