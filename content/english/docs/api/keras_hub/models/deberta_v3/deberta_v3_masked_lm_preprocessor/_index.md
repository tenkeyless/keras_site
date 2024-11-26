---
title: DebertaV3MaskedLMPreprocessor layer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/deberta_v3/deberta_v3_masked_lm_preprocessor.py#L14" >}}

### `DebertaV3MaskedLMPreprocessor` class

```python
keras_hub.models.DebertaV3MaskedLMPreprocessor(
    tokenizer,
    sequence_length=512,
    truncate="round_robin",
    mask_selection_rate=0.15,
    mask_selection_length=96,
    mask_token_rate=0.8,
    random_token_rate=0.1,
    **kwargs
)
```

DeBERTa preprocessing for the masked language modeling task.

This preprocessing layer will prepare inputs for a masked language modeling
task. It is primarily intended for use with the
[`keras_hub.models.DebertaV3MaskedLM`]({{< relref "/docs/api/keras_hub/models/deberta_v3/deberta_v3_masked_lm#debertav3maskedlm-class" >}}) task model. Preprocessing will occur in
multiple steps.

- Tokenize any number of input segments using the `tokenizer`.
- Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and
  `"<pad>"` tokens, i.e., adding a single `"<s>"` at the start of the
  entire sequence, `"</s></s>"` between each segment,
  and a `"</s>"` at the end of the entire sequence.
- Randomly select non-special tokens to mask, controlled by
  `mask_selection_rate`.
- Construct a `(x, y, sample_weight)` tuple suitable for training with a
  [`keras_hub.models.DebertaV3MaskedLM`]({{< relref "/docs/api/keras_hub/models/deberta_v3/deberta_v3_masked_lm#debertav3maskedlm-class" >}}) task model.

**Arguments**

- **tokenizer**: A `keras_hub.models.DebertaV3Tokenizer` instance.
- **sequence_length**: The length of the packed inputs.
- **mask_selection_rate**: The probability an input token will be dynamically
  masked.
- **mask_selection_length**: The maximum number of masked tokens supported
  by the layer.
- **mask_token_rate**: float. `mask_token_rate` must be
  between 0 and 1 which indicates how often the mask_token is
  substituted for tokens selected for masking.
  Defaults to `0.8`.
- **random_token_rate**: float. `random_token_rate` must be
  between 0 and 1 which indicates how often a random token is
  substituted for tokens selected for masking.
  Note: mask_token_rate + random_token_rate <= 1, and for
  (1 - mask_token_rate - random_token_rate), the token will not be
  changed. Defaults to `0.1`.
- **truncate**: string. The algorithm to truncate a list of batched segments
  to fit within `sequence_length`. The value can be either
  `round_robin` or `waterfall`:
  - `"round_robin"`: Available space is assigned one token at a
    time in a round-robin fashion to the inputs that still need
    some, until the limit is reached.
  - `"waterfall"`: The allocation of the budget is done using a
    "waterfall" algorithm that allocates quota in a
    left-to-right manner and fills up the buckets until we run
    out of budget. It supports an arbitrary number of segments.

**Examples**

Directly calling the layer on data.

```python
preprocessor = keras_hub.models.DebertaV3MaskedLMPreprocessor.from_preset(
    "deberta_v3_base_en"
)
# Tokenize and mask a single sentence.
preprocessor("The quick brown fox jumped.")
# Tokenize and mask a batch of single sentences.
preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
# Tokenize and mask sentence pairs.
# In this case, always convert input to tensors before calling the layer.
first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
second = tf.constant(["The fox tripped.", "Oh look, a whale."])
preprocessor((first, second))
```

Mapping with [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

```python
preprocessor = keras_hub.models.DebertaV3MaskedLMPreprocessor.from_preset(
    "deberta_v3_base_en"
)
first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
second = tf.constant(["The fox tripped.", "Oh look, a whale."])
# Map single sentences.
ds = tf.data.Dataset.from_tensor_slices(first)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Map sentence pairs.
ds = tf.data.Dataset.from_tensor_slices((first, second))
# Watch out for tf.data's default unpacking of tuples here!
# Best to invoke the `preprocessor` directly in this case.
ds = ds.map(
    lambda first, second: preprocessor(x=(first, second)),
    num_parallel_calls=tf.data.AUTOTUNE,
)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
DebertaV3MaskedLMPreprocessor.from_preset(
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

| Preset name               | Parameters | Description                                                                                                  |
| ------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| deberta_v3_extra_small_en | 70.68M     | 12-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText. |
| deberta_v3_small_en       | 141.30M    | 6-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText.  |
| deberta_v3_base_en        | 183.83M    | 12-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText. |
| deberta_v3_large_en       | 434.01M    | 24-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText. |
| deberta_v3_base_multi     | 278.22M    | 12-layer DeBERTaV3 model where case is maintained. Trained on the 2.5TB multilingual CC100 dataset.          |

### `tokenizer` property

```python
keras_hub.models.DebertaV3MaskedLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
