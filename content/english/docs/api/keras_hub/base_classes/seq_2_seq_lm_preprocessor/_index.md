---
title: Seq2SeqLMPreprocessor
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/seq_2_seq_lm_preprocessor.py#L15" >}}

### `Seq2SeqLMPreprocessor` class

```python
keras_hub.models.Seq2SeqLMPreprocessor(
    tokenizer, encoder_sequence_length=1024, decoder_sequence_length=1024, **kwargs
)
```

Base class for seq2seq language modeling preprocessing layers.

`Seq2SeqLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
create a preprocessing layer for seq2seq language modeling tasks. It is
intended to be paired with a `keras.models.Seq2SeqLM` task.

All `Seq2SeqLMPreprocessor` layers take inputs a dictionary input with keys
`"encoder_text"` and `"decoder_text"`.

This layer will always output a `(x, y, sample_weight)` tuple, where `x`
is a dictionary with the tokenized inputs, `y` contains the tokens from `x`
offset by 1, and `sample_weight` marks where `y` contains padded
values. The exact contents of `x` will vary depending on the model being
used.

a `Seq2SeqLMPreprocessor` contains two extra methods, `generate_preprocess`
and `generate_postprocess` for use with generation. See examples below.

All `Seq2SeqLMPreprocessor` tasks include a `from_preset()` constructor
which can be used to load a pre-trained config and vocabularies. You can
call the `from_preset()` constructor directly on this base class, in which
case the correct class for you model will be automatically instantiated.

Examples.

```python
preprocessor = keras_hub.models.Seq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=256,
    decoder_sequence_length=256,
)
# Tokenize, mask and pack a single sentence.
x = {
    "encoder_text": "The fox was sleeping.",
    "decoder_text": "The fox was awake.",
}
x, y, sample_weight = preprocessor(x)
# Tokenize and pad/truncate a batch of labeled sentences.
x = {
    "encoder_text": ["The fox was sleeping."],
    "decoder_text": ["The fox was awake."],
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
Seq2SeqLMPreprocessor.from_preset(preset, config_file="preprocessor.json", **kwargs)
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

| Preset name       | Parameters | Description                                                                                             |
| ----------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| bart_base_en      | 139.42M    | 6-layer BART model where case is maintained. Trained on BookCorpus, English Wikipedia and CommonCrawl.  |
| bart_large_en     | 406.29M    | 12-layer BART model where case is maintained. Trained on BookCorpus, English Wikipedia and CommonCrawl. |
| bart_large_en_cnn | 406.29M    | The `bart_large_en` backbone model fine-tuned on the CNN+DM summarization dataset.                      |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L222" >}}

### `save_to_preset` method

```python
Seq2SeqLMPreprocessor.save_to_preset(preset_dir)
```

Save preprocessor to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

### `tokenizer` property

```python
keras_hub.models.Seq2SeqLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
