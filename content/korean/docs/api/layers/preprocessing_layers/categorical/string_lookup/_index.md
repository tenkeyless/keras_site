---
title: StringLookup layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/string_lookup.py#L10" >}}

### `StringLookup` class

```python
keras.layers.StringLookup(
    max_tokens=None,
    num_oov_indices=1,
    mask_token=None,
    oov_token="[UNK]",
    vocabulary=None,
    idf_weights=None,
    invert=False,
    output_mode="int",
    pad_to_max_tokens=False,
    sparse=False,
    encoding="utf-8",
    name=None,
    **kwargs
)
```

A preprocessing layer that maps strings to (possibly encoded) indices.

This layer translates a set of arbitrary strings into integer output via a table-based vocabulary lookup. This layer will perform no splitting or transformation of input strings. For a layer that can split and tokenize natural language, see the [`keras.layers.TextVectorization`]({{< relref "/docs/api/layers/preprocessing_layers/text/text_vectorization#textvectorization-class" >}}) layer.

The vocabulary for the layer must be either supplied on construction or learned via `adapt()`. During `adapt()`, the layer will analyze a data set, determine the frequency of individual strings tokens, and create a vocabulary from them. If the vocabulary is capped in size, the most frequent tokens will be used to create the vocabulary and all others will be treated as out-of-vocabulary (OOV).

There are two possible output modes for the layer. When `output_mode` is `"int"`, input strings are converted to their index in the vocabulary (an integer). When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`, input strings are encoded into an array where each dimension corresponds to an element in the vocabulary.

The vocabulary can optionally contain a mask token as well as an OOV token (which can optionally occupy multiple indices in the vocabulary, as set by `num_oov_indices`). The position of these tokens in the vocabulary is fixed. When `output_mode` is `"int"`, the vocabulary will begin with the mask token (if set), followed by OOV indices, followed by the rest of the vocabulary. When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will begin with OOV indices and instances of the mask token will be dropped.

**Note:** This layer uses TensorFlow internally. It cannot be used as part of the compiled computation graph of a model with any backend other than TensorFlow. It can however be used with any backend when running eagerly. It can also always be used as part of an input preprocessing pipeline with any backend (outside the model itself), which is how we recommend to use this layer.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Arguments**

- **max_tokens**: Maximum size of the vocabulary for this layer. This should only be specified when adapting the vocabulary or when setting `pad_to_max_tokens=True`. If None, there is no cap on the size of the vocabulary. Note that this size includes the OOV and mask tokens. Defaults to `None`.
- **num_oov_indices**: The number of out-of-vocabulary tokens to use. If this value is more than 1, OOV inputs are modulated to determine their OOV value. If this value is 0, OOV inputs will cause an error when calling the layer. Defaults to `1`.
- **mask_token**: A token that represents masked inputs. When `output_mode` is `"int"`, the token is included in vocabulary and mapped to index 0. In other output modes, the token will not appear in the vocabulary and instances of the mask token in the input will be dropped. If set to `None`, no mask term will be added. Defaults to `None`.
- **oov_token**: Only used when `invert` is True. The token to return for OOV indices. Defaults to `"[UNK]"`.
- **vocabulary**: Optional. Either an array of integers or a string path to a text file. If passing an array, can pass a tuple, list, 1D NumPy array, or 1D tensor containing the integer vocbulary terms. If passing a file path, the file should contain one line per term in the vocabulary. If this argument is set, there is no need to `adapt()` the layer.
- **vocabulary_dtype**: The dtype of the vocabulary terms, for example `"int64"` or `"int32"`. Defaults to `"int64"`.
- **idf_weights**: Only valid when `output_mode` is `"tf_idf"`. A tuple, list, 1D NumPy array, or 1D tensor or the same length as the vocabulary, containing the floating point inverse document frequency weights, which will be multiplied by per sample term counts for the final TF-IDF weight. If the `vocabulary` argument is set, and `output_mode` is `"tf_idf"`, this argument must be supplied.
- **invert**: Only valid when `output_mode` is `"int"`. If `True`, this layer will map indices to vocabulary items instead of mapping vocabulary items to indices. Defaults to `False`.
- **output_mode**: Specification for the output of the layer. Values can be `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"` configuring the layer as follows:
  - `"int"`: Return the vocabulary indices of the input tokens.
  - `"one_hot"`: Encodes each individual element in the input into an array the same size as the vocabulary, containing a 1 at the element index. If the last dimension is size 1, will encode on that dimension. If the last dimension is not size 1, will append a new dimension for the encoded output.
  - `"multi_hot"`: Encodes each sample in the input into a single array the same size as the vocabulary, containing a 1 for each vocabulary term present in the sample. Treats the last dimension as the sample dimension, if input shape is `(..., sample_length)`, output shape will be `(..., num_tokens)`.
  - `"count"`: As `"multi_hot"`, but the int array contains a count of the number of times the token at that index appeared in the sample.
  - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to find the value in each token slot. For `"int"` output, any shape of input and output is supported. For all other output modes, currently only output up to rank 2 is supported. Defaults to `"int"`.
- **pad_to_max_tokens**: Only applicable when `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`. If `True`, the output will have its feature axis padded to `max_tokens` even if the number of unique tokens in the vocabulary is less than `max_tokens`, resulting in a tensor of shape `(batch_size, max_tokens)` regardless of vocabulary size. Defaults to `False`.
- **sparse**: Boolean. Only applicable to `"multi_hot"`, `"count"`, and `"tf_idf"` output modes. Only supported with TensorFlow backend. If `True`, returns a `SparseTensor` instead of a dense `Tensor`. Defaults to `False`.
- **encoding**: Optional. The text encoding to use to interpret the input strings. Defaults to `"utf-8"`.

**Examples**

**Creating a lookup layer with a known vocabulary**

This example creates a lookup layer with a pre-existing vocabulary.

```console
>>> vocab = ["a", "b", "c", "d"]
>>> data = [["a", "c", "d"], ["d", "z", "b"]]
>>> layer = StringLookup(vocabulary=vocab)
>>> layer(data)
array([[1, 3, 4],
       [4, 0, 2]])
```

**Creating a lookup layer with an adapted vocabulary**

This example creates a lookup layer and generates the vocabulary by analyzing the dataset.

```console
>>> data = [["a", "c", "d"], ["d", "z", "b"]]
>>> layer = StringLookup()
>>> layer.adapt(data)
>>> layer.get_vocabulary()
['[UNK]', 'd', 'z', 'c', 'b', 'a']
```

Note that the OOV token `"[UNK]"` has been added to the vocabulary. The remaining tokens are sorted by frequency (`"d"`, which has 2 occurrences, is first) then by inverse sort order.

```console
>>> data = [["a", "c", "d"], ["d", "z", "b"]]
>>> layer = StringLookup()
>>> layer.adapt(data)
>>> layer(data)
array([[5, 3, 1],
       [1, 2, 4]])
```

**Lookups with multiple OOV indices**

This example demonstrates how to use a lookup layer with multiple OOV indices. When a layer is created with more than one OOV index, any OOV values are hashed into the number of OOV buckets, distributing OOV values in a deterministic fashion across the set.

```console
>>> vocab = ["a", "b", "c", "d"]
>>> data = [["a", "c", "d"], ["m", "z", "b"]]
>>> layer = StringLookup(vocabulary=vocab, num_oov_indices=2)
>>> layer(data)
array([[2, 4, 5],
       [0, 1, 3]])
```

Note that the output for OOV value 'm' is 0, while the output for OOV value `"z"` is 1. The in-vocab terms have their output index increased by 1 from earlier examples (a maps to 2, etc) in order to make space for the extra OOV value.

**One-hot output**

Configure the layer with `output_mode='one_hot'`. Note that the first `num_oov_indices` dimensions in the ont_hot encoding represent OOV values.

```console
>>> vocab = ["a", "b", "c", "d"]
>>> data = ["a", "b", "c", "d", "z"]
>>> layer = StringLookup(vocabulary=vocab, output_mode='one_hot')
>>> layer(data)
array([[0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0.]], dtype=int64)
```

**Multi-hot output**

Configure the layer with `output_mode='multi_hot'`. Note that the first `num_oov_indices` dimensions in the multi_hot encoding represent OOV values.

```console
>>> vocab = ["a", "b", "c", "d"]
>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]
>>> layer = StringLookup(vocabulary=vocab, output_mode='multi_hot')
>>> layer(data)
array([[0., 1., 0., 1., 1.],
       [1., 0., 1., 0., 1.]], dtype=int64)
```

**Token count output**

Configure the layer with `output_mode='count'`. As with multi_hot output, the first `num_oov_indices` dimensions in the output represent OOV values.

```console
>>> vocab = ["a", "b", "c", "d"]
>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]
>>> layer = StringLookup(vocabulary=vocab, output_mode='count')
>>> layer(data)
array([[0., 1., 0., 1., 2.],
       [2., 0., 1., 0., 1.]], dtype=int64)
```

**TF-IDF output**

Configure the layer with `output_mode="tf_idf"`. As with multi_hot output, the first `num_oov_indices` dimensions in the output represent OOV values.

Each token bin will output `token_count * idf_weight`, where the idf weights are the inverse document frequency weights per token. These should be provided along with the vocabulary. Note that the `idf_weight` for OOV values will default to the average of all idf weights passed in.

```console
>>> vocab = ["a", "b", "c", "d"]
>>> idf_weights = [0.25, 0.75, 0.6, 0.4]
>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]
>>> layer = StringLookup(output_mode="tf_idf")
>>> layer.set_vocabulary(vocab, idf_weights=idf_weights)
>>> layer(data)
array([[0.  , 0.25, 0.  , 0.6 , 0.8 ],
       [1.0 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)
```

To specify the idf weights for oov values, you will need to pass the entire vocabulary including the leading oov token.

```console
>>> vocab = ["[UNK]", "a", "b", "c", "d"]
>>> idf_weights = [0.9, 0.25, 0.75, 0.6, 0.4]
>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]
>>> layer = StringLookup(output_mode="tf_idf")
>>> layer.set_vocabulary(vocab, idf_weights=idf_weights)
>>> layer(data)
array([[0.  , 0.25, 0.  , 0.6 , 0.8 ],
       [1.8 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)
```

When adapting the layer in `"tf_idf"` mode, each input sample will be considered a document, and IDF weight per token will be calculated as `log(1 + num_documents / (1 + token_document_count))`.

**Inverse lookup**

This example demonstrates how to map indices to strings using this layer. (You can also use `adapt()` with `inverse=True`, but for simplicity we'll pass the vocab in this example.)

```console
>>> vocab = ["a", "b", "c", "d"]
>>> data = [[1, 3, 4], [4, 0, 2]]
>>> layer = StringLookup(vocabulary=vocab, invert=True)
>>> layer(data)
array([[b'a', b'c', b'd'],
       [b'd', b'[UNK]', b'b']], dtype=object)
```

Note that the first index correspond to the oov token by default.

**Forward and inverse lookup pairs**

This example demonstrates how to use the vocabulary of a standard lookup layer to create an inverse lookup layer.

```console
>>> vocab = ["a", "b", "c", "d"]
>>> data = [["a", "c", "d"], ["d", "z", "b"]]
>>> layer = StringLookup(vocabulary=vocab)
>>> i_layer = StringLookup(vocabulary=vocab, invert=True)
>>> int_data = layer(data)
>>> i_layer(int_data)
array([[b'a', b'c', b'd'],
       [b'd', b'[UNK]', b'b']], dtype=object)
```

In this example, the input value `"z"` resulted in an output of `"[UNK]"`, since 1000 was not in the vocabulary - it got represented as an OOV, and all OOV values are returned as `"[UNK]"` in the inverse layer. Also, note that for the inverse to work, you must have already set the forward layer vocabulary either directly or via `adapt()` before calling `get_vocabulary()`.
