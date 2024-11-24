---
title: preprocessor
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L15" >}}

### `Preprocessor` class

`keras_hub.models.Preprocessor()`

Base class for preprocessing layers.

A `Preprocessor` layer provides a complete preprocessing setup for a given task. It handles tokenization, audio/image conversion, and any other necessary preprocessing steps.

This class can be subclassed similar to any [`keras.layers.Layer`](/api/layers/base_layer#layer-class), by defining `build()`, `call()` and `get_config()` methods. All subclasses should set the `tokenizer` or `audio_converter` or `image_converter` properties during construction as needed.

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

`Preprocessor.from_preset(preset, config_file="preprocessor.json", **kwargs)`

Instantiate a [`keras_hub.models.Preprocessor`](/api/keras_hub/base_classes/preprocessor#preprocessor-class) from a model preset.

A preset is a directory of configs, weights and other file assets used to save and load a pre-trained model. The `preset` can be passed as one of:

1.  a built-in preset identifier like `'bert_base_en'`
2.  a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3.  a Hugging Face handle like `'hf://user/bert_base_en'`
4.  a path to a local preset directory like `'./bert_base_en'`

For any `Preprocessor` subclass, you can run `cls.presets.keys()` to list all built-in presets available on the class.

As there are usually multiple preprocessing classes for a given model, this method should be called on a specific subclass like `keras_hub.models.BertTextClassifierPreprocessor.from_preset()`.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models handle, a Hugging Face handle, or a path to a local directory.

**Examples**

`# Load a preprocessor for Gemma generation. preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(     "gemma_2b_en", )  # Load a preprocessor for Bert classification. preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(     "bert_base_en", )`

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L222" >}}

### `save_to_preset` method

`Preprocessor.save_to_preset(preset_dir)`

Save preprocessor to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

---

### `tokenizer` property

`keras_hub.models.Preprocessor.tokenizer`

The tokenizer used to tokenize strings.

---
