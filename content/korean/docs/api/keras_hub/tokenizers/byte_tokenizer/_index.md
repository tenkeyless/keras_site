---
title: ByteTokenizer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_tokenizer.py#L23" >}}

### `ByteTokenizer` class

```python
keras_hub.tokenizers.ByteTokenizer(
    lowercase=True,
    sequence_length=None,
    normalization_form=None,
    errors="replace",
    replacement_char=65533,
    dtype="int32",
    **kwargs
)
```

Raw byte tokenizer.

This tokenizer is a vocabulary-free tokenizer which will tokenize text as
as raw bytes from [0, 256).

Tokenizer outputs can either be padded and truncated with a
`sequence_length` argument, or left un-truncated. The exact output will
depend on the rank of the input tensors.

If input is a batch of strings:
By default, the layer will output a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last
dimension of the output is ragged. If `sequence_length` is set, the layer
will output a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) where all inputs have been padded or
truncated to `sequence_length`.

If input is a scalar string:
There are two cases here. If `sequence_length` is set, the output will be
a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) of shape `[sequence_length]`. Otherwise, the output will
be a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) of shape `[None]`.

The output dtype can be controlled via the
`dtype` argument, which should be an integer type
("int16", "int32", etc.).

**Arguments**

- **lowercase**: boolean. If True, the input text will be converted to
  lowercase before tokenization.
- **sequence_length**: int. If set, the output will be converted to a dense
  tensor and padded/trimmed so all outputs are of sequence_length.
- **normalization_form**: string. One of the following values: (None, "NFC",
  "NFKC", "NFD", "NFKD"). If set, every UTF-8 string in the input
  tensor text will be normalized to the given form before tokenizing.
- **errors**: One of ('replace', 'remove', 'strict'). Specifies the
  `detokenize()` behavior when an invalid tokenizer is encountered.
  The value of `'strict'` will cause the operation to produce a
  `InvalidArgument` error on any invalid input formatting. A value of
  `'replace'` will cause the tokenizer to replace any invalid
  formatting in the input with the `replacement_char` codepoint.
  A value of `'ignore'` will cause the tokenizer to skip any invalid
  formatting in the input and produce no corresponding output
  character.
- **replacement_char**: int. The replacement character to
  use when an invalid byte sequence is encountered and when `errors`
  is set to "replace" (same behaviour as
  https://www.tensorflow.org/api\_docs/python/tf/strings/unicode\_transcode).
  (U+FFFD) is `65533`. Defaults to `65533`.

**Examples**

Basic usage.

```console
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
>>> outputs = tokenizer("hello")
>>> np.array(outputs)
array([104, 101, 108, 108, 111], dtype=int32)
```

Ragged outputs.

```console
>>> inputs = ["hello", "hi"]
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
>>> seq1, seq2 = tokenizer(inputs)
>>> np.array(seq1)
array([104, 101, 108, 108, 111])
>>> np.array(seq2)
array([104, 105])
```

Dense outputs.

```console
>>> inputs = ["hello", "hi"]
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer(sequence_length=8)
>>> seq1, seq2 = tokenizer(inputs)
>>> np.array(seq1)
array([104, 101, 108, 108, 111,   0,   0,   0], dtype=int32)
>>> np.array(seq2)
array([104, 105,   0,   0,   0,   0,   0,   0], dtype=int32)
```

Tokenize, then batch for ragged outputs.

```console
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
>>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
>>> ds = ds.map(tokenizer)
>>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
>>> ds.take(1).get_single_element()
<tf.RaggedTensor [[104, 101, 108, 108, 111], [102, 117, 110]]>
```

Batch, then tokenize for ragged outputs.

```console
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
>>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
>>> ds = ds.batch(2).map(tokenizer)
>>> ds.take(1).get_single_element()
<tf.RaggedTensor [[104, 101, 108, 108, 111], [102, 117, 110]]>
```

Tokenize, then batch for dense outputs (`sequence_length` provided).

```console
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer(sequence_length=5)
>>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
>>> ds = ds.map(tokenizer)
>>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
>>> ds.take(1).get_single_element()
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[104, 101, 108, 108, 111],
       [102, 117, 110,   0,   0]], dtype=int32)>
```

Batch, then tokenize for dense outputs. (`sequence_length` provided).

```console
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer(sequence_length=5)
>>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
>>> ds = ds.batch(2).map(tokenizer)
>>> ds.take(1).get_single_element()
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[104, 101, 108, 108, 111],
       [102, 117, 110,   0,   0]], dtype=int32)>
```

Detokenization.

```console
>>> inputs = [104, 101, 108, 108, 111]
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
>>> tokenizer.detokenize(inputs)
'hello'
```

Detokenization with invalid bytes.

```console
>>> # The 255 below is invalid utf-8.
>>> inputs = [104, 101, 255, 108, 108, 111]
>>> tokenizer = keras_hub.tokenizers.ByteTokenizer(
...     errors="replace", replacement_char=88)
>>> tokenizer.detokenize(inputs)
'heXllo'
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_tokenizer.py#L201" >}}

### `tokenize` method

```python
ByteTokenizer.tokenize(inputs)
```

Transform input tensors of strings into output tokens.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_tokenizer.py#L232" >}}

### `detokenize` method

```python
ByteTokenizer.detokenize(inputs)
```

Transform tokens back into strings.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_tokenizer.py#L195" >}}

### `get_vocabulary` method

```python
ByteTokenizer.get_vocabulary()
```

Get the tokenizer vocabulary as a list of strings terms.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_tokenizer.py#L191" >}}

### `vocabulary_size` method

```python
ByteTokenizer.vocabulary_size()
```

Get the integer size of the tokenizer vocabulary.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_tokenizer.py#L264" >}}

### `token_to_id` method

```python
ByteTokenizer.token_to_id(token)
```

Convert a string token to an integer id.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_tokenizer.py#L255" >}}

### `id_to_token` method

```python
ByteTokenizer.id_to_token(id)
```

Convert an integer id to a string token.
