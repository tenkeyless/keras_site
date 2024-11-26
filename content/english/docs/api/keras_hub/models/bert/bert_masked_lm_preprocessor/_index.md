---
title: BertMaskedLMPreprocessor layer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/bert/bert_masked_lm_preprocessor.py#L7" >}}

### `BertMaskedLMPreprocessor` class

```python
keras_hub.models.BertMaskedLMPreprocessor(
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

BERT preprocessing for the masked language modeling task.

This preprocessing layer will prepare inputs for a masked language modeling
task. It is primarily intended for use with the
[`keras_hub.models.BertMaskedLM`]({{< relref "/docs/api/keras_hub/models/bert/bert_masked_lm#bertmaskedlm-class" >}}) task model. Preprocessing will occur in
multiple steps.

1. Tokenize any number of input segments using the `tokenizer`.
2. Pack the inputs together with the appropriate `"[CLS]"`, `"[SEP]"` and
   `"[PAD]"` tokens.
3. Randomly select non-special tokens to mask, controlled by
   `mask_selection_rate`.
4. Construct a `(x, y, sample_weight)` tuple suitable for training with a
   [`keras_hub.models.BertMaskedLM`]({{< relref "/docs/api/keras_hub/models/bert/bert_masked_lm#bertmaskedlm-class" >}}) task model.

**Arguments**

- **tokenizer**: A `keras_hub.models.BertTokenizer` instance.
- **sequence_length**: int. The length of the packed inputs.
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
- **mask_selection_rate**: float. The probability an input token will be
  dynamically masked.
- **mask_selection_length**: int. The maximum number of masked tokens
  in a given sample.
- **mask_token_rate**: float. The probability the a selected token will be
  replaced with the mask token.
- **random_token_rate**: float. The probability the a selected token will be
  replaced with a random token from the vocabulary. A selected token
  will be left as is with probability
  `1 - mask_token_rate - random_token_rate`.

**Call arguments**

- **x**: A tensor of single string sequences, or a tuple of multiple
  tensor sequences to be packed together. Inputs may be batched or
  unbatched. For single sequences, raw python inputs will be converted
  to tensors. For multiple sequences, pass tensors directly.
- **y**: Label data. Should always be `None` as the layer generates labels.
- **sample_weight**: Label weights. Should always be `None` as the layer
  generates label weights.

**Examples**

Directly calling the layer on data.

```python
preprocessor = keras_hub.models.BertMaskedLMPreprocessor.from_preset(
    "bert_base_en_uncased"
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
preprocessor = keras_hub.models.BertMaskedLMPreprocessor.from_preset(
    "bert_base_en_uncased"
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
BertMaskedLMPreprocessor.from_preset(
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

| Preset name               | Parameters | Description                                                                                     |
| ------------------------- | ---------- | ----------------------------------------------------------------------------------------------- |
| bert_tiny_en_uncased      | 4.39M      | 2-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.   |
| bert_small_en_uncased     | 28.76M     | 4-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.   |
| bert_medium_en_uncased    | 41.37M     | 8-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.   |
| bert_base_en_uncased      | 109.48M    | 12-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.  |
| bert_base_en              | 108.31M    | 12-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus.       |
| bert_base_zh              | 102.27M    | 12-layer BERT model. Trained on Chinese Wikipedia.                                              |
| bert_base_multi           | 177.85M    | 12-layer BERT model where case is maintained. Trained on trained on Wikipedias of 104 languages |
| bert_large_en_uncased     | 335.14M    | 24-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.  |
| bert_large_en             | 333.58M    | 24-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus.       |
| bert_tiny_en_uncased_sst2 | 4.39M      | The bert_tiny_en_uncased backbone model fine-tuned on the SST-2 sentiment analysis dataset.     |

### `tokenizer` property

```python
keras_hub.models.BertMaskedLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
