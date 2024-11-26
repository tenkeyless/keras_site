---
title: BytePairTokenizer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_pair_tokenizer.py#L198" >}}

### `BytePairTokenizer` class

```python
keras_nlp.tokenizers.BytePairTokenizer(
    vocabulary=None,
    merges=None,
    sequence_length=None,
    add_prefix_space=False,
    unsplittable_tokens=None,
    dtype="int32",
    **kwargs
)
```

Bype-pair encoding tokenizer layer.

This BPE tokenizer provides the same functionality as the official GPT-2
tokenizer. Given the same `vocabulary` which maps tokens to ids, and `merges`
which describes BPE merge rules, it should provide the same output
as OpenAI implementation (https://github.com/openai/gpt-2/blob/master/src/encoder.py).
Different from OpenAI, this implementation is graph-compatible, so you can
use it within a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline.

If input is a batch of strings (rank > 0):
By default, the layer will output a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last
dimension of the output is ragged. If `sequence_length` is set, the layer
will output a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) where all inputs have been padded or
truncated to `sequence_length`.
If input is a scalar string (rank == 0):
By default, the layer will output a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) with static shape
`[None]`. If `sequence_length` is set, the output will be
a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) of shape `[sequence_length]`.

**Arguments**

- **vocabulary**: string or dict, maps token to integer ids. If it is a
  string, it should be the file path to a json file.
- **merges**: string or list, contains the merge rule. If it is a string,
  it should be the file path to merge rules. The merge rule file
  should have one merge rule per line.
- **sequence_length**: int. If set, the output will be
  padded or truncated to the `sequence_length`. Defaults to `None`.
- **add_prefix_space**: bool. Whether to add an
  initial space to the input. This tokenizer is whitespace aware,
  and will tokenize a word with a leading space differently. Adding
  a prefix space to the first word will cause it to be tokenized
  equivalently to all subsequent words in the sequence.
  Defaults to `False`.
- **unsplittable_tokens**: list. A list of strings that will
  never be split during the word-level splitting applied before the
  byte-pair encoding. This can be used to ensure special tokens map to
  unique indices in the vocabulary, even if these special tokens
  contain splittable characters such as punctuation. Special tokens
  must still be included in `vocabulary`. Defaults to `None`.

**Examples**

Tokenize

```console
>>> vocab = {"butter": 1, "fly": 2}
>>> merge = ["b u", "t t", "e r", "bu tt", "butt er", "f l", "fl y"]
>>> tokenizer = keras_hub.tokenizers.BytePairTokenizer(vocab, merge)
>>> outputs = tokenizer("butterfly")
>>> np.array(outputs)
array([1, 2], dtype=int32)
>>> seq1, seq2 = tokenizer(["butterfly", "butter"])
>>> np.array(seq1)
array([1, 2])
>>> np.array(seq2)
array([1])
>>> tokenizer = keras_hub.tokenizers.BytePairTokenizer(
...     vocab, merge, sequence_length=2)
>>> seq1, seq2 = tokenizer(["butterfly", "butter"])
>>> np.array(seq1)
array([1, 2], dtype=int32)
>>> np.array(seq2)
array([1, 0], dtype=int32)
```

Detokenize

```console
>>> vocab = {"butter": 1, "fly": 2}
>>> merge = ["b u", "t t", "e r", "bu tt", "butt er", "f l", "fl y"]
>>> tokenizer = keras_hub.tokenizers.BytePairTokenizer(vocab, merge)
>>> tokenizer.detokenize([[1, 2]])
['butterfly']
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_pair_tokenizer.py#L527" >}}

### `tokenize` method

```python
BytePairTokenizer.tokenize(inputs)
```

Transform input tensors of strings into output tokens.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_pair_tokenizer.py#L592" >}}

### `detokenize` method

```python
BytePairTokenizer.detokenize(inputs)
```

Transform tokens back into strings.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_pair_tokenizer.py#L387" >}}

### `get_vocabulary` method

```python
BytePairTokenizer.get_vocabulary()
```

Get the tokenizer vocabulary as a list of strings tokens.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_pair_tokenizer.py#L392" >}}

### `vocabulary_size` method

```python
BytePairTokenizer.vocabulary_size()
```

Get the integer size of the tokenizer vocabulary.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_pair_tokenizer.py#L410" >}}

### `token_to_id` method

```python
BytePairTokenizer.token_to_id(token)
```

Convert a string token to an integer id.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/byte_pair_tokenizer.py#L397" >}}

### `id_to_token` method

```python
BytePairTokenizer.id_to_token(id)
```

Convert an integer id to a string token.
