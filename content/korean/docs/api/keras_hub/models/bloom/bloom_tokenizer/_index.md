---
title: BloomTokenizer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/bloom/bloom_tokenizer.py#L6" >}}

### `BloomTokenizer` class

```python
keras_hub.tokenizers.BloomTokenizer(vocabulary=None, merges=None, **kwargs)
```

A BLOOM tokenizer using Byte-Pair Encoding subword segmentation.

This tokenizer class will tokenize raw strings into integer sequences and
is based on [`keras_hub.tokenizers.BytePairTokenizer`]({{< relref "/docs/api/keras_hub/tokenizers/byte_pair_tokenizer#bytepairtokenizer-class" >}}). Unlike the
underlying tokenizer, it will check for all special tokens needed by BLOOM
models and provides a `from_preset()` method to automatically download
a matching vocabulary for a BLOOM preset.

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
tokenizer = keras_hub.models.BloomTokenizer.from_preset("bloom_560m_multi")
tokenizer("The quick brown fox jumped.")

# Batched input.
tokenizer(["The quick brown fox jumped.", "The fox slept."])

# Detokenization.
tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

# Custom vocabulary.
vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "a": 3, "Ġquick": 4, "Ġfox": 5}
merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
merges += ["Ġ f", "o x", "Ġf ox"]
tokenizer = keras_hub.models.BloomTokenizer(vocabulary=vocab, merges=merges)
tokenizer("a quick fox.")
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L213" >}}

### `from_preset` method

```python
BloomTokenizer.from_preset(preset, config_file="tokenizer.json", **kwargs)
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

| Preset name       | Parameters | Description                                                                                                       |
| ----------------- | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| bloom_560m_multi  | 559.21M    | 24-layer Bloom model with hidden dimension of 1024. trained on 45 natural languages and 12 programming languages. |
| bloom_1.1b_multi  | 1.07B      | 24-layer Bloom model with hidden dimension of 1536. trained on 45 natural languages and 12 programming languages. |
| bloom_1.7b_multi  | 1.72B      | 24-layer Bloom model with hidden dimension of 2048. trained on 45 natural languages and 12 programming languages. |
| bloom_3b_multi    | 3.00B      | 30-layer Bloom model with hidden dimension of 2560. trained on 45 natural languages and 12 programming languages. |
| bloomz_560m_multi | 559.21M    | 24-layer Bloom model with hidden dimension of 1024. finetuned on crosslingual task mixture (xP3) dataset.         |
| bloomz_1.1b_multi | 1.07B      | 24-layer Bloom model with hidden dimension of 1536. finetuned on crosslingual task mixture (xP3) dataset.         |
| bloomz_1.7b_multi | 1.72B      | 24-layer Bloom model with hidden dimension of 2048. finetuned on crosslingual task mixture (xP3) dataset.         |
| bloomz_3b_multi   | 3.00B      | 30-layer Bloom model with hidden dimension of 2560. finetuned on crosslingual task mixture (xP3) dataset.         |
