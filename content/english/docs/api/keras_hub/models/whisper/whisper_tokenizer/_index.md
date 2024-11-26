---
title: WhisperTokenizer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/whisper/whisper_tokenizer.py#L15" >}}

### `WhisperTokenizer` class

```python
keras_hub.tokenizers.WhisperTokenizer(
    vocabulary=None, merges=None, special_tokens=None, language_tokens=None, **kwargs
)
```

Whisper text tokenizer using Byte-Pair Encoding subword segmentation.

This tokenizer class will tokenize raw strings into integer sequences and
is based on [`keras_hub.tokenizers.BytePairTokenizer`]({{< relref "/docs/api/keras_hub/tokenizers/byte_pair_tokenizer#bytepairtokenizer-class" >}}).
This tokenizer does not provide truncation or padding of inputs.

**Arguments**

- **vocabulary**: string or dict, maps token to integer ids. If it is a
  string, it should be the file path to a json file.
- **merges**: string or list, contains the merge rule. If it is a string,
  it should be the file path to merge rules. The merge rule file
  should have one merge rule per line. Every merge rule contains
  merge entities separated by a space.
- **special_tokens**: string or dict, maps special tokens to integer IDs. If
  it is a string, it should be the path to a JSON file.
- **language_tokens**: string or dict, maps language tokens to integer IDs. If
  not None, the tokenizer will be assumed to be a multilingual
  tokenizer.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L213" >}}

### `from_preset` method

```python
WhisperTokenizer.from_preset(preset, config_file="tokenizer.json", **kwargs)
```

Instantiate a `keras_hub.models.Tokenizer` from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Tokenizer` subclass, you can run `cls.presets.keys()` to list
all built-in presets available on the class.

This constructor can be called in one of two ways. Either from the base
class like `keras_hub.models.Tokenizer.from_preset()`, or from
a model class like `keras_hub.models.GemmaTokenizer.from_preset()`.
If calling from the base class, the subclass of the returning object
will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the
  model architecture. If `False`, the weights will be randomly
  initialized.

**Examples**

```python
# Load a preset tokenizer.
tokenizer = keras_hub.tokenizer.Tokenizer.from_preset("bert_base_en")
# Tokenize some input.
tokenizer("The quick brown fox tripped.")
# Detokenize some input.
tokenizer.detokenize([5, 6, 7, 8, 9])
```

| Preset name            | Parameters | Description                                                                                                                                 |
| ---------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| whisper_tiny_en        | 37.18M     | 4-layer Whisper model. Trained on 438,000 hours of labelled English speech data.                                                            |
| whisper_base_en        | 124.44M    | 6-layer Whisper model. Trained on 438,000 hours of labelled English speech data.                                                            |
| whisper_small_en       | 241.73M    | 12-layer Whisper model. Trained on 438,000 hours of labelled English speech data.                                                           |
| whisper_medium_en      | 763.86M    | 24-layer Whisper model. Trained on 438,000 hours of labelled English speech data.                                                           |
| whisper_tiny_multi     | 37.76M     | 4-layer Whisper model. Trained on 680,000 hours of labelled multilingual speech data.                                                       |
| whisper_base_multi     | 72.59M     | 6-layer Whisper model. Trained on 680,000 hours of labelled multilingual speech data.                                                       |
| whisper_small_multi    | 241.73M    | 12-layer Whisper model. Trained on 680,000 hours of labelled multilingual speech data.                                                      |
| whisper_medium_multi   | 763.86M    | 24-layer Whisper model. Trained on 680,000 hours of labelled multilingual speech data.                                                      |
| whisper_large_multi    | 1.54B      | 32-layer Whisper model. Trained on 680,000 hours of labelled multilingual speech data.                                                      |
| whisper_large_multi_v2 | 1.54B      | 32-layer Whisper model. Trained for 2.5 epochs on 680,000 hours of labelled multilingual speech data. An improved of `whisper_large_multi`. |
