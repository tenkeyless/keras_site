---
title: GemmaCausalLMPreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/gemma/gemma_causal_lm_preprocessor.py#L7" >}}

### `GemmaCausalLMPreprocessor` class

```python
keras_nlp.models.GemmaCausalLMPreprocessor(
    tokenizer, sequence_length=1024, add_start_token=True, add_end_token=True, **kwargs
)
```

Gemma Causal LM preprocessor.

This preprocessing layer is meant for use with
[`keras_hub.models.GemmaCausalLM`]({{< relref "/docs/api/keras_hub/models/gemma/gemma_causal_lm#gemmacausallm-class" >}}). By default, it will take in batches of
strings, and return outputs in a `(x, y, sample_weight)` format, where the
`y` label is the next token id in the `x` sequence.

For use with generation, the layer also exposes two methods
`generate_preprocess()` and `generate_postprocess()`. When this preprocessor
is attached to a [`keras_hub.models.GemmaCausalLM`]({{< relref "/docs/api/keras_hub/models/gemma/gemma_causal_lm#gemmacausallm-class" >}}) instance, these methods
will be called implicitly in `generate()`. They can also be called
standalone (e.g. to precompute preprocessing inputs for generation in a
separate process).

**Arguments**

- **tokenizer**: A `keras_hub.models.GemmaTokenizer` instance.
- **sequence_length**: The length of the packed inputs.
- **add_start_token**: If `True`, the preprocessor will prepend the tokenizer
  start token to each input sequence.
- **add_end_token**: If `True`, the preprocessor will append the tokenizer
  end token to each input sequence.

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
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en"
)
# Tokenize and pack a single sentence.
preprocessor("The quick brown fox jumped.")
# Tokenize a batch of sentences.
preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
# Apply tokenization to a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
features = tf.constant(["The quick brown fox.", "Call me Ishmael."])
ds = tf.data.Dataset.from_tensor_slices(features)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Prepare tokens for generation (no end token).
preprocessor.generate_preprocess(["The quick brown fox jumped."])
# Map generation outputs back to strings.
preprocessor.generate_postprocess({
    'token_ids': np.array([[2, 714, 4320, 8426, 25341, 32292, 235265, 0]]),
    'padding_mask': np.array([[ 1,  1,  1,  1,  1,  1,  1, 0]]),
})
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
GemmaCausalLMPreprocessor.from_preset(
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

| Preset name                   | Parameters | Description                                                                                                                                                                |
| ----------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| gemma_2b_en                   | 2.51B      | 2 billion parameter, 18-layer, base Gemma model.                                                                                                                           |
| gemma_instruct_2b_en          | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model.                                                                                                              |
| gemma_1.1_instruct_2b_en      | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                       |
| code_gemma_1.1_2b_en          | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion. The 1.1 update improves model quality. |
| code_gemma_2b_en              | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                        |
| gemma_7b_en                   | 8.54B      | 7 billion parameter, 28-layer, base Gemma model.                                                                                                                           |
| gemma_instruct_7b_en          | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model.                                                                                                              |
| gemma_1.1_instruct_7b_en      | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                       |
| code_gemma_7b_en              | 8.54B      | 7 billion parameter, 28-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                        |
| code_gemma_instruct_7b_en     | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code.                                          |
| code_gemma_1.1_instruct_7b_en | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code. The 1.1 update improves model quality.   |
| gemma2_2b_en                  | 2.61B      | 2 billion parameter, 26-layer, base Gemma model.                                                                                                                           |
| gemma2_instruct_2b_en         | 2.61B      | 2 billion parameter, 26-layer, instruction tuned Gemma model.                                                                                                              |
| gemma2_9b_en                  | 9.24B      | 9 billion parameter, 42-layer, base Gemma model.                                                                                                                           |
| gemma2_instruct_9b_en         | 9.24B      | 9 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                              |
| gemma2_27b_en                 | 27.23B     | 27 billion parameter, 42-layer, base Gemma model.                                                                                                                          |
| gemma2_instruct_27b_en        | 27.23B     | 27 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                             |
| shieldgemma_2b_en             | 2.61B      | 2 billion parameter, 26-layer, ShieldGemma model.                                                                                                                          |
| shieldgemma_9b_en             | 9.24B      | 9 billion parameter, 42-layer, ShieldGemma model.                                                                                                                          |
| shieldgemma_27b_en            | 27.23B     | 27 billion parameter, 42-layer, ShieldGemma model.                                                                                                                         |

### `tokenizer` property

```python
keras_nlp.models.GemmaCausalLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
