---
title: UnicodeCodepointTokenizer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/unicode_codepoint_tokenizer.py#L15" >}}

### `UnicodeCodepointTokenizer` class

```python
keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    sequence_length=None,
    lowercase=True,
    normalization_form=None,
    errors="replace",
    replacement_char=65533,
    input_encoding="UTF-8",
    output_encoding="UTF-8",
    vocabulary_size=None,
    dtype="int32",
    **kwargs
)
```

A unicode character tokenizer layer.

This tokenizer is a vocabulary free tokenizer which tokenizes text as
unicode character codepoints.

Tokenizer outputs can either be padded and truncated with a
`sequence_length` argument, or left un-truncated. The exact output will
depend on the rank of the input tensors.

If input is a batch of strings (rank > 0):
By default, the layer will output a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last
dimension of the output is ragged. If `sequence_length` is set, the layer
will output a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) where all inputs have been padded or
truncated to `sequence_length`.

If input is a scalar string (rank == 0):
By default, the layer will output a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) with static shape
`[None]`. If `sequence_length` is set, the output will be
a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) of shape `[sequence_length]`.

The output dtype can be controlled via the `dtype` argument, which should be
an integer type ("int16", "int32", etc.).

**Arguments**

- **lowercase**: If `True`, the input text will be first lowered before
  tokenization.
- **sequence_length**: If set, the output will be converted to a dense
  tensor and padded/trimmed so all outputs are of sequence_length.
- **normalization_form**: One of the following string values (None, 'NFC',
  'NFKC', 'NFD', 'NFKD'). If set will normalize unicode to the given
  form before tokenizing.
- **errors**: One of ('replace', 'remove', 'strict'). Specifies the
  `detokenize()` behavior when an invalid codepoint is encountered.
  The value of `'strict'` will cause the tokenizer to produce a
  `InvalidArgument` error on any invalid input formatting. A value of
  `'replace'` will cause the tokenizer to replace any invalid
  formatting in the input with the replacement_char codepoint.
  A value of `'ignore'` will cause the tokenizer to skip any invalid
  formatting in the input and produce no corresponding output
  character.
- **replacement_char**: The unicode codepoint to use in place of invalid
  codepoints. (U+FFFD) is `65533`. Defaults to `65533`.
- **input_encoding**: One of ("UTF-8", "UTF-16-BE", or "UTF-32-BE").
  One of The encoding of the input text. Defaults to `"UTF-8"`.
- **output_encoding**: One of ("UTF-8", "UTF-16-BE", or "UTF-32-BE").
  The encoding of the output text. Defaults to `"UTF-8"`.
- **vocabulary_size**: Set the vocabulary `vocabulary_size`,
  by clamping all codepoints to the range [0, vocabulary_size).
  Effectively this will make the `vocabulary_size - 1` id the
  the OOV value.

**Examples**

Basic Usage.

```console
>>> inputs = "Unicode Tokenizer"
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer()
>>> outputs = tokenizer(inputs)
>>> np.array(outputs)
array([117, 110, 105,  99, 111, 100, 101,  32, 116, 111, 107, 101, 110,
    105, 122, 101, 114], dtype=int32)
```

Ragged outputs.

```console
>>> inputs = ["à¤ªà¥à¤¸à¥à¤¤à¤", "Ú©ØªØ§Ø¨"]
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer()
>>> seq1, seq2 = tokenizer(inputs)
>>> np.array(seq1)
array([2346, 2369, 2360, 2381, 2340, 2325])
>>> np.array(seq2)
array([1705, 1578, 1575, 1576])
```

Dense outputs.

```console
>>> inputs = ["à¤ªà¥à¤¸à¥à¤¤à¤", "Ú©ØªØ§Ø¨"]
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer(
...     sequence_length=8)
>>> seq1, seq2 = tokenizer(inputs)
>>> np.array(seq1)
array([2346, 2369, 2360, 2381, 2340, 2325,    0,    0], dtype=int32)
>>> np.array(seq2)
array([1705, 1578, 1575, 1576,    0,    0,    0,    0], dtype=int32)
```

Tokenize, then batch for ragged outputs.

```console
>>> inputs = ["Book", "à¤ªà¥à¤¸à¥à¤¤à¤", "Ú©ØªØ§Ø¨"]
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer()
>>> ds = tf.data.Dataset.from_tensor_slices(inputs)
>>> ds = ds.map(tokenizer)
>>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(3))
>>> ds.take(1).get_single_element()
<tf.RaggedTensor [[98, 111, 111, 107],
    [2346, 2369, 2360, 2381, 2340, 2325],
    [1705, 1578, 1575, 1576]]>
```

Batch, then tokenize for ragged outputs.

```console
>>> inputs = ["Book", "à¤ªà¥à¤¸à¥à¤¤à¤", "Ú©ØªØ§Ø¨"]
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer()
>>> ds = tf.data.Dataset.from_tensor_slices(inputs)
>>> ds = ds.batch(3).map(tokenizer)
>>> ds.take(1).get_single_element()
<tf.RaggedTensor [[98, 111, 111, 107],
    [2346, 2369, 2360, 2381, 2340, 2325],
    [1705, 1578, 1575, 1576]]>
```

Tokenize, then batch for dense outputs (`sequence_length` provided).

```console
>>> inputs = ["Book", "à¤ªà¥à¤¸à¥à¤¤à¤", "Ú©ØªØ§Ø¨"]
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer(
...     sequence_length=5)
>>> ds = tf.data.Dataset.from_tensor_slices(inputs)
>>> ds = ds.map(tokenizer)
>>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(3))
>>> ds.take(1).get_single_element()
<tf.Tensor: shape=(3, 5), dtype=int32, numpy=
array([[  98,  111,  111,  107,    0],
    [2346, 2369, 2360, 2381, 2340],
    [1705, 1578, 1575, 1576,    0]], dtype=int32)>
```

Batch, then tokenize for dense outputs (`sequence_length` provided).

```console
>>> inputs = ["Book", "à¤ªà¥à¤¸à¥à¤¤à¤", "Ú©ØªØ§Ø¨"]
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer(
...     sequence_length=5)
>>> ds = tf.data.Dataset.from_tensor_slices(inputs)
>>> ds = ds.batch(3).map(tokenizer)
>>> ds.take(1).get_single_element()
<tf.Tensor: shape=(3, 5), dtype=int32, numpy=
array([[  98,  111,  111,  107,    0],
    [2346, 2369, 2360, 2381, 2340],
    [1705, 1578, 1575, 1576,    0]], dtype=int32)>
```

Tokenization with truncation.

```console
>>> inputs = ["I Like to Travel a Lot", "à¤®à¥à¤ à¤à¤¿à¤¤à¤¾à¤¬à¥à¤ à¤ªà¤¢à¤¼à¤¨à¤¾ à¤ªà¤¸à¤à¤¦ à¤à¤°à¤¤à¤¾ à¤¹à¥à¤"]
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer(
...     sequence_length=5)
>>> outputs = tokenizer(inputs)
>>> np.array(outputs)
array([[ 105,   32,  108,  105,  107],
       [2350, 2376, 2306,   32, 2325]], dtype=int32)
```

Tokenization with vocabulary_size.

```console
>>> latin_ext_cutoff = 592
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer(
...     vocabulary_size=latin_ext_cutoff)
>>> outputs = tokenizer("Â¿CÃ³mo estÃ¡s?")
>>> np.array(outputs)
array([191,  99, 243, 109, 111,  32, 101, 115, 116, 225, 115,  63],
      dtype=int32)
>>> outputs = tokenizer("à¤à¤ª à¤à¥à¤¸à¥ à¤¹à¥à¤")
>>> np.array(outputs)
array([591, 591,  32, 591, 591, 591, 591,  32, 591, 591, 591],
      dtype=int32)
```

Detokenization.

```console
>>> inputs = tf.constant([110, 105, 110, 106,  97], dtype="int32")
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer()
>>> tokenizer.detokenize(inputs)
'ninja'
```

Detokenization with padding.

```console
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer(
...     sequence_length=7)
>>> dataset = tf.data.Dataset.from_tensor_slices(["a b c", "b c", "a"])
>>> dataset = dataset.map(tokenizer)
>>> dataset.take(1).get_single_element()
<tf.Tensor: shape=(7,), dtype=int32,
    numpy=array([97, 32, 98, 32, 99,  0,  0], dtype=int32)>
>>> detokunbatched = dataset.map(tokenizer.detokenize)
>>> detokunbatched.take(1).get_single_element()
<tf.Tensor: shape=(), dtype=string, numpy=b'a b c'>
```

Detokenization with invalid bytes.

```console
>>> inputs = tf.constant([110, 105, 10000000, 110, 106,  97])
>>> tokenizer = keras_hub.tokenizers.UnicodeCodepointTokenizer(
...     errors="replace", replacement_char=88)
>>> tokenizer.detokenize(inputs)
'niXnja'
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/unicode_codepoint_tokenizer.py#L272" >}}

### `tokenize` method

```python
UnicodeCodepointTokenizer.tokenize(inputs)
```

Transform input tensors of strings into output tokens.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/unicode_codepoint_tokenizer.py#L309" >}}

### `detokenize` method

```python
UnicodeCodepointTokenizer.detokenize(inputs)
```

Transform tokens back into strings.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/unicode_codepoint_tokenizer.py#L266" >}}

### `get_vocabulary` method

```python
UnicodeCodepointTokenizer.get_vocabulary()
```

Get the tokenizer vocabulary as a list of strings terms.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/unicode_codepoint_tokenizer.py#L261" >}}

### `vocabulary_size` method

```python
UnicodeCodepointTokenizer.vocabulary_size()
```

Get the size of the tokenizer vocabulary. None implies no vocabulary
size was provided

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/unicode_codepoint_tokenizer.py#L332" >}}

### `token_to_id` method

```python
UnicodeCodepointTokenizer.token_to_id(token)
```

Convert a string token to an integer id.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/unicode_codepoint_tokenizer.py#L323" >}}

### `id_to_token` method

```python
UnicodeCodepointTokenizer.id_to_token(id)
```

Convert an integer id to a string token.
