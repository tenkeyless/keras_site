---
title: compute_sentence_piece_proto function
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer_trainer.py#L19" >}}

### `compute_sentence_piece_proto` function

```python
keras_hub.tokenizers.compute_sentence_piece_proto(
    data, vocabulary_size, model_type="unigram", proto_output_file=None, lowercase=False
)
```

A utility to train a SentencePiece vocabulary.

Trains a SentencePiece vocabulary from an input dataset or a list of
filenames.

If `data` is a list of filenames, the file format is required to be plain
text files, and the text will be read in line by line during training.

**Arguments**

- **data**: A [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), or a list of filenames.
- **vocabulary_size**: int. The maximum size of a vocabulary to be trained.
- **model_type**: str. The model algorithm must be one of
  `"unigram"`, `"bpe"`, `"word"` or `"char"`. Defaults to `"unigram"`.
- **proto_output_file**: str. If provided it will be used
  as model_file which is passed to model_writer.
  If `None`, the model_file will be `io.BytesIO` object.
  Defaults to `None`.
- **lowercase**: bool. If True, the input text will be
  lowercased before tokenization. Defaults to `False`.

**Returns**

A `bytes` object with a serialized SentencePiece proto or
`None` if proto_output_file if provided.

**Examples**

Basic Usage (from Dataset).

```console
>>> inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
>>> proto = keras_hub.tokenizers.compute_sentence_piece_proto(inputs, vocabulary_size=15)
>>> tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(proto=proto)
>>> outputs = inputs.map(tokenizer)
>>> for output in outputs:
...     print(output)
tf.Tensor([ 4  8 12  5  9 14  5  6 13  4  7 10 11  6 13],
shape=(15,), dtype=int32)
```

Basic Usage (with files).

```python
with open("test.txt", "w+") as f: f.write("Drifting Along\n")
inputs = ["test.txt"]
proto = keras_hub.tokenizers.compute_sentence_piece_proto(
     inputs, vocabulary_size=15, proto_output_file="model.spm")
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(proto="model.spm")
ds = tf.data.Dataset.from_tensor_slices(["the quick brown fox."])
ds = ds.map(tokenizer)
```

Usage with lowercase

```console
>>> inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
>>> proto = keras_hub.tokenizers.compute_sentence_piece_proto(
...     inputs, vocabulary_size=15, lowercase=True)
>>> tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(proto=proto)
>>> outputs = inputs.map(tokenizer)
>>> for output in outputs:
...     print(output)
tf.Tensor([ 4  8 12  5  9 14  5  6 13  4  7 10 11  6 13],
shape=(15,), dtype=int32)
```
