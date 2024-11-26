---
title: AlbertTokenizer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/albert/albert_tokenizer.py#L8" >}}

### `AlbertTokenizer` class

```python
keras_nlp.tokenizers.AlbertTokenizer(proto, **kwargs)
```

ALBERT tokenizer layer based on SentencePiece.

This tokenizer class will tokenize raw strings into integer sequences and
is based on [`keras_hub.tokenizers.SentencePieceTokenizer`]({{< relref "/docs/api/keras_hub/tokenizers/sentence_piece_tokenizer#sentencepiecetokenizer-class" >}}). Unlike the
underlying tokenizer, it will check for all special tokens needed by
ALBERT models and provides a `from_preset()` method to automatically
download a matching vocabulary for a ALBERT preset.

If input is a batch of strings (rank > 0), the layer will output a
[`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last dimension of the output is ragged.

If input is a scalar string (rank == 0), the layer will output a dense
[`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) with static shape `[None]`.

**Arguments**

- **proto**: Either a `string` path to a SentencePiece proto file, or a
  `bytes` object with a serialized SentencePiece proto. See the
  [SentencePiece repository](https://github.com/google/sentencepiece)
  for more details on the format.

**Examples**

```python
# Unbatched input.
tokenizer = keras_hub.models.AlbertTokenizer.from_preset(
    "albert_base_en_uncased",
)
tokenizer("The quick brown fox jumped.")
# Batched input.
tokenizer(["The quick brown fox jumped.", "The fox slept."])
# Detokenization.
tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
# Custom vocabulary.
bytes_io = io.BytesIO()
ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
sentencepiece.SentencePieceTrainer.train(
    sentence_iterator=ds.as_numpy_iterator(),
    model_writer=bytes_io,
    vocab_size=10,
    model_type="WORD",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="[CLS]",
    eos_piece="[SEP]",
    user_defined_symbols="[MASK]",
)
tokenizer = keras_hub.models.AlbertTokenizer(
    proto=bytes_io.getvalue(),
)
tokenizer("The quick brown fox jumped.")
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L213" >}}

### `from_preset` method

```python
AlbertTokenizer.from_preset(preset, config_file="tokenizer.json", **kwargs)
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

| Preset name                         | Parameters | Description                                                                                      |
| ----------------------------------- | ---------- | ------------------------------------------------------------------------------------------------ |
| albert_base_en_uncased              | 11.68M     | 12-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| albert_large_en_uncased             | 17.68M     | 24-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| albert_extra_large_en_uncased       | 58.72M     | 24-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
| albert_extra_extra_large_en_uncased | 222.60M    | 12-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus. |
