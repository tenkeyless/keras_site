---
title: Phi3CausalLMPreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/phi3/phi3_causal_lm_preprocessor.py#L7" >}}

### `Phi3CausalLMPreprocessor` class

```python
keras_hub.models.Phi3CausalLMPreprocessor(
    tokenizer, sequence_length=1024, add_start_token=True, add_end_token=True, **kwargs
)
```

Phi3 Causal LM preprocessor.

This preprocessing layer is meant for use with
[`keras_hub.models.Phi3CausalLM`]({{< relref "/docs/api/keras_hub/models/phi3/phi3_causal_lm#phi3causallm-class" >}}). By default, it will take in batches of
strings, and return outputs in a `(x, y, sample_weight)` format, where the
`y` label is the next token id in the `x` sequence.

For use with generation, the layer also exposes two methods
`generate_preprocess()` and `generate_postprocess()`. When this preprocessor
is attached to a [`keras_hub.models.Phi3CausalLM`]({{< relref "/docs/api/keras_hub/models/phi3/phi3_causal_lm#phi3causallm-class" >}}) instance, these methods
will be called implicitly in `generate()`. They can also be called
standalone (e.g. to precompute preprocessing inputs for generation in a
separate process).

**Arguments**

- **tokenizer**: A `keras_hub.models.Phi3Tokenizer` instance.
- **sequence_length**: The length of the packed inputs.
- **add_start_token**: If `True`, the preprocessor will prepend the tokenizer
  start token to each input sequence. Default is `True`.
- **add_end_token**: If `True`, the preprocessor will append the tokenizer
  end token to each input sequence. Default is `False`.

**Call arguments**

- **x**: A string, [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) or list of python strings.
- **y**: Label data. Should always be `None` as the layer generates labels.
- **sample_weight**: Label weights. Should always be `None` as the layer
  generates label weights.
- **sequence_length**: Pass to override the configured `sequence_length` of
  the layer.

**Examples**

```python
# Load the preprocessor from a preset.
preprocessor = keras_hub.models.Phi3CausalLMPreprocessor.from_preset(
    "phi3_mini_4k_instruct_en"
)
# Tokenize and pack a single sentence.
sentence = tf.constant("League of legends")
preprocessor(sentence)
# Same output.
preprocessor("League of legends")
# Tokenize a batch of sentences.
sentences = tf.constant(["Taco tuesday", "Fish taco please!"])
preprocessor(sentences)
# Same output.
preprocessor(["Taco tuesday", "Fish taco please!"])
# Map a dataset to preprocess a single sentence.
features = tf.constant(
    [
        "Avatar 2 is amazing!",
        "Well, I am not sure.",
    ]
)
labels = tf.constant([1, 0])
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Map a dataset to preprocess unlabled sentences.
ds = tf.data.Dataset.from_tensor_slices(features)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
Phi3CausalLMPreprocessor.from_preset(
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

| Preset name                | Parameters | Description                                                                                                                                                                                                                                                                   |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| phi3_mini_4k_instruct_en   | 3.82B      | 3.8 billion parameters, 32 layers, 4k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties.   |
| phi3_mini_128k_instruct_en | 3.82B      | 3.8 billion parameters, 32 layers, 128k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties. |

### `tokenizer` property

```python
keras_hub.models.Phi3CausalLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
