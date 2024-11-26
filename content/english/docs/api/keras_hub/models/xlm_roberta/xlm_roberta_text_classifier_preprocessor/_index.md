---
title: XLMRobertaTextClassifierPreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/xlm_roberta/xlm_roberta_text_classifier_preprocessor.py#L19" >}}

### `XLMRobertaTextClassifierPreprocessor` class

```python
keras_hub.models.XLMRobertaTextClassifierPreprocessor(
    tokenizer, sequence_length=512, truncate="round_robin", **kwargs
)
```

An XLM-RoBERTa preprocessing layer which tokenizes and packs inputs.

This preprocessing layer will do three things:

1. Tokenize any number of input segments using the `tokenizer`.
2. Pack the inputs together using a [`keras_hub.layers.MultiSegmentPacker`]({{< relref "/docs/api/keras_hub/preprocessing_layers/multi_segment_packer#multisegmentpacker-class" >}}).
   with the appropriate `"<s>"`, `"</s>"` and `"<pad>"` tokens, i.e., adding
   a single `"<s>"` at the start of the entire sequence, `"</s></s>"` at the
   end of each segment, save the last and a `"</s>"` at the end of the
   entire sequence.
3. Construct a dictionary with keys `"token_ids"` and `"padding_mask"`,
   that can be passed directly to an XLM-RoBERTa model.

This layer can be used directly with [`tf.data.Dataset.map`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) to preprocess
string data in the `(x, y, sample_weight)` format used by
[`keras.Model.fit`]({{< relref "/docs/api/models/model_training_apis#fit-method" >}}).

**Arguments**

- **tokenizer**: A [`keras_hub.tokenizers.XLMRobertaTokenizer`]({{< relref "/docs/api/keras_hub/models/xlm_roberta/xlm_roberta_tokenizer#xlmrobertatokenizer-class" >}}) instance.
- **sequence_length**: The length of the packed inputs.
- **truncate**: The algorithm to truncate a list of batched segments to fit
  within `sequence_length`. The value can be either `round_robin` or
  `waterfall`:
  - `"round_robin"`: Available space is assigned one token at a
    time in a round-robin fashion to the inputs that still need
    some, until the limit is reached.
  - `"waterfall"`: The allocation of the budget is done using a
    "waterfall" algorithm that allocates quota in a
    left-to-right manner and fills up the buckets until we run
    out of budget. It supports an arbitrary number of segments.

**Call arguments**

- **x**: A tensor of single string sequences, or a tuple of multiple
  tensor sequences to be packed together. Inputs may be batched or
  unbatched. For single sequences, raw python inputs will be converted
  to tensors. For multiple sequences, pass tensors directly.
- **y**: Any label data. Will be passed through unaltered.
- **sample_weight**: Any label weight data. Will be passed through unaltered.

**Examples**

Directly calling the layer on data.

```python
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
    "xlm_roberta_base_multi"
)
# Tokenize and pack a single sentence.
preprocessor("The quick brown fox jumped.")
# Tokenize a batch of single sentences.
preprocessor(["The quick brown fox jumped.", "Ø§Ø³ÙÙ Ø§Ø³ÙØ§Ø¹ÙÙ"])
# Preprocess a batch of sentence pairs.
# When handling multiple sequences, always convert to tensors first!
first = tf.constant(["The quick brown fox jumped.", "Ø§Ø³ÙÙ Ø§Ø³ÙØ§Ø¹ÙÙ"])
second = tf.constant(["The fox tripped.", "Ø§ÙØ£Ø³Ø¯ ÙÙÙ Ø§ÙØºØ§Ø¨Ø©"])
preprocessor((first, second))
# Custom vocabulary.
def train_sentencepiece(ds, vocab_size):
    bytes_io = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=vocab_size,
        model_type="WORD",
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    return bytes_io.getvalue()
ds = tf.data.Dataset.from_tensor_slices(
    ["the quick brown fox", "the earth is round"]
)
proto = train_sentencepiece(ds, vocab_size=10)
tokenizer = keras_hub.models.XLMRobertaTokenizer(proto=proto)
preprocessor = keras_hub.models.XLMRobertaTextClassifierPreprocessor(
    tokenizer
)
preprocessor("The quick brown fox jumped.")
```

Mapping with [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

```python
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
    "xlm_roberta_base_multi"
)
first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
second = tf.constant(["The fox tripped.", "Oh look, a whale."])
label = tf.constant([1, 1])
# Map labeled single sentences.
ds = tf.data.Dataset.from_tensor_slices((first, label))
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Map unlabeled single sentences.
ds = tf.data.Dataset.from_tensor_slices(first)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Map labeled sentence pairs.
ds = tf.data.Dataset.from_tensor_slices(((first, second), label))
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Map unlabeled sentence pairs.
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
XLMRobertaTextClassifierPreprocessor.from_preset(
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

| Preset name             | Parameters | Description                                                                                   |
| ----------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| xlm_roberta_base_multi  | 277.45M    | 12-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages. |
| xlm_roberta_large_multi | 558.84M    | 24-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages. |

### `tokenizer` property

```python
keras_hub.models.XLMRobertaTextClassifierPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
