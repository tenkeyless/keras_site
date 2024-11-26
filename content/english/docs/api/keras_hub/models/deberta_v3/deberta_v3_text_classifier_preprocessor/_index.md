---
title: DebertaV3TextClassifierPreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/deberta_v3/deberta_v3_text_classifier_preprocessor.py#L16" >}}

### `DebertaV3TextClassifierPreprocessor` class

```python
keras_hub.models.DebertaV3TextClassifierPreprocessor(
    tokenizer, sequence_length=512, truncate="round_robin", **kwargs
)
```

A DeBERTa preprocessing layer which tokenizes and packs inputs.

This preprocessing layer will do three things:

- Tokenize any number of input segments using the `tokenizer`.
- Pack the inputs together using a [`keras_hub.layers.MultiSegmentPacker`]({{< relref "/docs/api/keras_hub/preprocessing_layers/multi_segment_packer#multisegmentpacker-class" >}}).
  with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
- Construct a dictionary with keys `"token_ids"` and `"padding_mask"`, that
  can be passed directly to a DeBERTa model.

This layer can be used directly with [`tf.data.Dataset.map`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) to preprocess
string data in the `(x, y, sample_weight)` format used by
[`keras.Model.fit`]({{< relref "/docs/api/models/model_training_apis#fit-method" >}}).

The call method of this layer accepts three arguments, `x`, `y`, and
`sample_weight`. `x` can be a python string or tensor representing a single
segment, a list of python strings representing a batch of single segments,
or a list of tensors representing multiple segments to be packed together.
`y` and `sample_weight` are both optional, can have any format, and will be
passed through unaltered.

Special care should be taken when using [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) to map over an unlabeled
tuple of string segments. [`tf.data.Dataset.map`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) will unpack this tuple
directly into the call arguments of this layer, rather than forward all
argument to `x`. To handle this case, it is recommended to explicitly call
the layer, e.g. `ds.map(lambda seg1, seg2: preprocessor(x=(seg1, seg2)))`.

**Arguments**

- **tokenizer**: A `keras_hub.models.DebertaV3Tokenizer` instance.
- **sequence_length**: The length of the packed inputs.
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
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
    "deberta_v3_base_en"
)
# Tokenize and pack a single sentence.
preprocessor("The quick brown fox jumped.")
# Tokenize a batch of single sentences.
preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
# Preprocess a batch of sentence pairs.
# When handling multiple sequences, always convert to tensors first!
first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])
second = tf.constant(["The fox tripped.", "Oh look, a whale."])
preprocessor((first, second))
# Custom vocabulary.
bytes_io = io.BytesIO()
ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
sentencepiece.SentencePieceTrainer.train(
    sentence_iterator=ds.as_numpy_iterator(),
    model_writer=bytes_io,
    vocab_size=9,
    model_type="WORD",
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
    pad_piece="[PAD]",
    bos_piece="[CLS]",
    eos_piece="[SEP]",
    unk_piece="[UNK]",
)
tokenizer = keras_hub.models.DebertaV3Tokenizer(
    proto=bytes_io.getvalue(),
)
preprocessor = keras_hub.models.DebertaV3TextClassifierPreprocessor(
    tokenizer
)
preprocessor("The quick brown fox jumped.")
```

Mapping with [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

```python
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
    "deberta_v3_base_en"
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
DebertaV3TextClassifierPreprocessor.from_preset(
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
keras_hub.models.DebertaV3TextClassifierPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
