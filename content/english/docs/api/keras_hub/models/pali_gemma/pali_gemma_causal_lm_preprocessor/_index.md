---
title: PaliGemmaCausalLMPreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/pali_gemma/pali_gemma_causal_lm_preprocessor.py#L20" >}}

### `PaliGemmaCausalLMPreprocessor` class

```python
keras_hub.models.PaliGemmaCausalLMPreprocessor(
    tokenizer,
    image_converter=None,
    sequence_length=1024,
    add_start_token=True,
    add_end_token=True,
    **kwargs
)
```

Base class for causal language modeling preprocessing layers.

`CausalLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
create a preprocessing layer for causal language modeling tasks. It is
intended to be paired with a `keras.models.CausalLM` task.

All `CausalLMPreprocessor` take inputs a single input. This can be a single
string or a batch of strings. See examples below. These inputs
will be tokenized and padded/truncated to a fixed sequence length.

This layer will always output a `(x, y, sample_weight)` tuple, where `x`
is a dictionary with the tokenized inputs, `y` contains the tokens from `x`
offset by 1, and `sample_weight` marks where `y` contains padded
values. The exact contents of `x` will vary depending on the model being
used.

a `CausalLMPreprocessor` contains two extra methods, `generate_preprocess`
and `generate_postprocess` for use with generation. See examples below.

All `CausalLMPreprocessor` tasks include a `from_preset()` constructor
which can be used to load a pre-trained config and vocabularies. You can
call the `from_preset()` constructor directly on this base class, in which
case the correct class for you model will be automatically instantiated.

Examples.

```python
preprocessor = keras_hub.models.CausalLMPreprocessor.from_preset(
    "bert_base_en_uncased",
    sequence_length=256, # Optional.
)
# Tokenize, mask and pack a single sentence.
x = "The quick brown fox jumped."
x, y, sample_weight = preprocessor(x)
# Tokenize and pad/truncate a batch of labeled sentences.
x = ["The quick brown fox jumped.", "Call me Ishmael."]
x, y, sample_weight = preprocessor(x)
# With a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
ds = tf.data.Dataset.from_tensor_slices(x)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Generate preprocess and postprocess.
x = preprocessor.generate_preprocess(x)  # Tokenized numeric inputs.
x = preprocessor.generate_postprocess(x)  # Detokenized string outputs.
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
PaliGemmaCausalLMPreprocessor.from_preset(
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

| Preset name           | Parameters | Description                                                 |
| --------------------- | ---------- | ----------------------------------------------------------- |
| pali_gemma_3b_mix_224 | 2.92B      | image size 224, mix fine tuned, text sequence length is 256 |
| pali_gemma_3b_mix_448 | 2.92B      | image size 448, mix fine tuned, text sequence length is 512 |
| pali_gemma_3b_224     | 2.92B      | image size 224, pre trained, text sequence length is 128    |
| pali_gemma_3b_448     | 2.92B      | image size 448, pre trained, text sequence length is 512    |
| pali_gemma_3b_896     | 2.93B      | image size 896, pre trained, text sequence length is 512    |

### `tokenizer` property

```python
keras_hub.models.PaliGemmaCausalLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
