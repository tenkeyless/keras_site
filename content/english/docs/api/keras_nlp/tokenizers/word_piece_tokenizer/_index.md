---
title: word_piece_tokenizer
toc: false
---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/word_piece_tokenizer.py#L190)

### `WordPieceTokenizer` class

`keras_nlp.tokenizers.WordPieceTokenizer(     vocabulary=None,     sequence_length=None,     lowercase=False,     strip_accents=False,     split=True,     split_on_cjk=True,     suffix_indicator="##",     oov_token="[UNK]",     special_tokens=None,     special_tokens_in_strings=False,     dtype="int32",     **kwargs )`

A WordPiece tokenizer layer.

This layer provides an efficient, in graph, implementation of the WordPiece algorithm used by BERT and other models.

To make this layer more useful out of the box, the layer will pre-tokenize the input, which will optionally lower-case, strip accents, and split the input on whitespace and punctuation. Each of these pre-tokenization steps is not reversible. The `detokenize` method will join words with a space, and will not invert `tokenize` exactly.

If a more custom pre-tokenization step is desired, the layer can be configured to apply only the strict WordPiece algorithm by passing `lowercase=False`, `strip_accents=False` and `split=False`. In this case, inputs should be pre-split string tensors or ragged tensors.

Tokenizer outputs can either be padded and truncated with a `sequence_length` argument, or left un-truncated. The exact output will depend on the rank of the input tensors.

If input is a batch of strings (rank > 0): By default, the layer will output a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last dimension of the output is ragged. If `sequence_length` is set, the layer will output a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) where all inputs have been padded or truncated to `sequence_length`.

If input is a scalar string (rank == 0): By default, the layer will output a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) with static shape `[None]`. If `sequence_length` is set, the output will be a dense [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) of shape `[sequence_length]`.

The output dtype can be controlled via the `dtype` argument, which should be either an integer or string type.

**Arguments**

- **vocabulary**: A list of strings or a string filename path. If passing a list, each element of the list should be a single WordPiece token string. If passing a filename, the file should be a plain text file containing a single WordPiece token per line.
- **sequence_length**: int. If set, the output will be converted to a dense tensor and padded/trimmed so all outputs are of sequence_length.
- **lowercase**: bool. If `True`, the input text will be lowercased before tokenization. Defaults to `False`.
- **strip_accents**: bool. If `True`, all accent marks will be removed from text before tokenization. Defaults to `False`.
- **split**: bool. If `True`, input will be split on whitespace and punctuation marks, and all punctuation marks will be kept as tokens. If `False`, input should be split ("pre-tokenized") before calling the tokenizer, and passed as a dense or ragged tensor of whole words. Defaults to `True`.
- **split_on_cjk**: bool. If True, input will be split on CJK characters, i.e., Chinese, Japanese, Korean and Vietnamese characters (https://en.wikipedia.org/wiki/CJK\_Unified\_Ideographs\_(Unicode\_block)). Note that this is applicable only when `split` is True. Defaults to `True`.
- **suffix_indicator**: str. The characters prepended to a WordPiece to indicate that it is a suffix to another subword. E.g. "##ing". Defaults to `"##"`.
- **oov_token**: str. The string value to substitute for an unknown token. It must be included in the vocab. Defaults to `"[UNK]"`.
- **special_tokens_in_strings**: bool. A bool to indicate if the tokenizer should expect special tokens in input strings that should be tokenized and mapped correctly to their ids. Defaults to False.

**References**

- [Schuster and Nakajima, 2012](https://research.google/pubs/pub37842/)
- [Song et al., 2020](https://arxiv.org/abs/2012.15524)

**Examples**

Ragged outputs.

`>>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."] >>> inputs = "The quick brown fox." >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer( ...     vocabulary=vocab, ...     lowercase=True, ... ) >>> outputs = tokenizer(inputs) >>> np.array(outputs) array([1, 2, 3, 4, 5, 6, 7], dtype=int32)`

Dense outputs.

`>>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."] >>> inputs = ["The quick brown fox."] >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer( ...     vocabulary=vocab, ...     sequence_length=10, ...     lowercase=True, ... ) >>> outputs = tokenizer(inputs) >>> np.array(outputs) array([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]], dtype=int32)`

String output.

`>>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."] >>> inputs = "The quick brown fox." >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer( ...     vocabulary=vocab, ...     lowercase=True, ...     dtype="string", ... ) >>> tokenizer(inputs) ['the', 'qu', '##ick', 'br', '##own', 'fox', '.']`

Detokenization.

`>>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."] >>> inputs = "The quick brown fox." >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer( ...     vocabulary=vocab, ...     lowercase=True, ... ) >>> tokenizer.detokenize(tokenizer.tokenize(inputs)) 'the quick brown fox .'`

Custom splitting.

`>>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."] >>> inputs = "The$quick$brown$fox" >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer( ...     vocabulary=vocab, ...     split=False, ...     lowercase=True, ...     dtype='string', ... ) >>> split_inputs = tf.strings.split(inputs, sep="$") >>> tokenizer(split_inputs) ['the', 'qu', '##ick', 'br', '##own', 'fox']`

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/word_piece_tokenizer.py#L459)

### `tokenize` method

`WordPieceTokenizer.tokenize(inputs)`

Transform input tensors of strings into output tokens.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/word_piece_tokenizer.py#L505)

### `detokenize` method

`WordPieceTokenizer.detokenize(inputs)`

Transform tokens back into strings.

**Arguments**

- **inputs**: Input tensor, or dict/list/tuple of input tensors.
- **\*args**: Additional positional arguments.
- **\*\*kwargs**: Additional keyword arguments.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/word_piece_tokenizer.py#L407)

### `get_vocabulary` method

`WordPieceTokenizer.get_vocabulary()`

Get the tokenizer vocabulary as a list of strings tokens.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/word_piece_tokenizer.py#L412)

### `vocabulary_size` method

`WordPieceTokenizer.vocabulary_size()`

Get the integer size of the tokenizer vocabulary.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/word_piece_tokenizer.py#L427)

### `token_to_id` method

`WordPieceTokenizer.token_to_id(token)`

Convert a string token to an integer id.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/word_piece_tokenizer.py#L417)

### `id_to_token` method

`WordPieceTokenizer.id_to_token(id)`

Convert an integer id to a string token.

---
