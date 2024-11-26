---
title: SentencePieceTokenizer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer.py#L32" >}}

### `SentencePieceTokenizer` class

```python
keras_hub.tokenizers.SentencePieceTokenizer(
    proto=None,
    sequence_length=None,
    dtype="int32",
    add_bos=False,
    add_eos=False,
    **kwargs
)
```

A SentencePiece tokenizer layer.

This layer provides an implementation of SentencePiece tokenization
as described in the [SentencePiece paper](https://arxiv.org/abs/1808.06226)
and the [SentencePiece package](https://pypi.org/project/sentencepiece/).
The tokenization will run entirely within the Tensorflow graph, and can
be saved inside a [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}).

By default, the layer will output a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last
dimension of the output is ragged after whitespace splitting and sub-word
tokenizing. If `sequence_length` is set, the layer will output a dense
[`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) where all inputs have been padded or truncated to
`sequence_length`. The output dtype can be controlled via the `dtype`
argument, which should be either an integer or string type.

**Arguments**

- **proto**: Either a `string` path to a SentencePiece proto file, or a
  `bytes` object with a serialized SentencePiece proto. See the
  [SentencePiece repository](https://github.com/google/sentencepiece)
  for more details on the format.
- **sequence_length**: If set, the output will be converted to a dense
  tensor and padded/trimmed so all outputs are of `sequence_length`.
- **add_bos**: Add beginning of sentence token to the result.
- **add_eos**: Add end of sentence token to the result. Token is always
  truncated if output is longer than specified `sequence_length`.

**References**

- [Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)

**Examples**

From bytes.

```python
def train_sentence_piece_bytes(ds, size):
    bytes_io = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=size,
    )
    return bytes_io.getvalue()
# Train a sentencepiece proto.
ds = tf.data.Dataset.from_tensor_slices(["the quick brown fox."])
proto = train_sentence_piece_bytes(ds, 20)
# Tokenize inputs.
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(proto=proto)
ds = ds.map(tokenizer)
```

From a file.

```python
def train_sentence_piece_file(ds, path, size):
    with open(path, "wb") as model_file:
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=ds.as_numpy_iterator(),
            model_writer=model_file,
            vocab_size=size,
        )
# Train a sentencepiece proto.
ds = tf.data.Dataset.from_tensor_slices(["the quick brown fox."])
proto = train_sentence_piece_file(ds, "model.spm", 20)
# Tokenize inputs.
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(proto="model.spm")
ds = ds.map(tokenizer)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer.py#L224" >}}

### `tokenize` method

```python
SentencePieceTokenizer.tokenize(inputs)
```

Transform input tensors of strings into output tokens.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer.py#L252" >}}

### `detokenize` method

```python
SentencePieceTokenizer.detokenize(inputs)
```

Transform tokens back into strings.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer.py#L181" >}}

### `get_vocabulary` method

```python
SentencePieceTokenizer.get_vocabulary()
```

Get the tokenizer vocabulary.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer.py#L176" >}}

### `vocabulary_size` method

```python
SentencePieceTokenizer.vocabulary_size()
```

Get the integer size of the tokenizer vocabulary.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer.py#L200" >}}

### `token_to_id` method

```python
SentencePieceTokenizer.token_to_id(token)
```

Convert a string token to an integer id.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/sentence_piece_tokenizer.py#L190" >}}

### `id_to_token` method

```python
SentencePieceTokenizer.id_to_token(id)
```

Convert an integer id to a string token.
