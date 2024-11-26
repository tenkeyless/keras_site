---
title: OPTTokenizer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/opt/opt_tokenizer.py#L6" >}}

### `OPTTokenizer` class

```python
keras_hub.tokenizers.OPTTokenizer(vocabulary=None, merges=None, **kwargs)
```

An OPT tokenizer using Byte-Pair Encoding subword segmentation.

This tokenizer class will tokenize raw strings into integer sequences and
is based on [`keras_hub.tokenizers.BytePairTokenizer`]({{< relref "/docs/api/keras_hub/tokenizers/byte_pair_tokenizer#bytepairtokenizer-class" >}}). Unlike the
underlying tokenizer, it will check for all special tokens needed by OPT
models and provides a `from_preset()` method to automatically download
a matching vocabulary for a OPT preset.

If input is a batch of strings (rank > 0), the layer will output a
[`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last dimension of the output is ragged.
If input is a scalar string (rank == 0), the layer will output a dense
[`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) with static shape `[None]`.

**Arguments**

- **vocabulary**: string or dict, maps token to integer ids. If it is a
  string, it should be the file path to a json file.
- **merges**: string or list, contains the merge rule. If it is a string,
  it should be the file path to merge rules. The merge rule file
  should have one merge rule per line. Every merge rule contains
  merge entities separated by a space.

**Examples**

```python
# Unbatched input.
tokenizer = keras_hub.models.OPTTokenizer.from_preset(
    "opt_125m_en",
)
tokenizer("The quick brown fox jumped.")
# Batched input.
tokenizer(["The quick brown fox jumped.", "The fox slept."])
# Detokenization.
tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
# Custom vocabulary.
vocab = {"<pad>": 1, "</s>": 2, "Ä quick": 4, "Ä fox": 5}
merges = ["Ä  q", "u i", "c k", "ui ck", "Ä q uick"]
merges += ["Ä  f", "o x", "Ä f ox"]
tokenizer = keras_hub.models.OPTTokenizer(vocabulary=vocab, merges=merges)
tokenizer("The quick brown fox jumped.")
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L213" >}}

### `from_preset` method

```python
OPTTokenizer.from_preset(preset, config_file="tokenizer.json", **kwargs)
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

| Preset name | Parameters | Description                                                                                                      |
| ----------- | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| opt_125m_en | 125.24M    | 12-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |
| opt_1.3b_en | 1.32B      | 24-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |
| opt_2.7b_en | 2.70B      | 32-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |
| opt_6.7b_en | 6.70B      | 32-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |
