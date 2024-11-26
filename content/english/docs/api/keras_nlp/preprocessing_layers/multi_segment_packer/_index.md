---
title: MultiSegmentPacker layer
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/preprocessing/multi_segment_packer.py#L16" >}}

### `MultiSegmentPacker` class

```python
keras_nlp.layers.MultiSegmentPacker(
    sequence_length,
    start_value,
    end_value,
    sep_value=None,
    pad_value=None,
    truncate="round_robin",
    **kwargs
)
```

Packs multiple sequences into a single fixed width model input.

This layer packs multiple input sequences into a single fixed width sequence
containing start and end delimeters, forming a dense input suitable for a
classification task for BERT and BERT-like models.

Takes as input a tuple of token segments. Each tuple element should contain
the tokens for a segment, passed as tensors, [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor)s, or lists.
For batched input, each element in the tuple of segments should be a list of
lists or a rank two tensor. For unbatched inputs, each element should be a
list or rank one tensor.

The layer will process inputs as follows:

- Truncate all input segments to fit within `sequence_length` according to
  the `truncate` strategy.
- Concatenate all input segments, adding a single `start_value` at the
  start of the entire sequence, and multiple `end_value`s at the end of
  each segment.
- Pad the resulting sequence to `sequence_length` using `pad_tokens`.
- Calculate a separate tensor of "segment ids", with integer type and the
  same shape as the packed token output, where each integer index of the
  segment the token originated from. The segment id of the `start_value`
  is always 0, and the segment id of each `end_value` is the segment that
  precedes it.

**Arguments**

- **sequence_length**: int. The desired output length.
- **start_value**: int/str/list/tuple. The id(s) or token(s) that are to be
  placed at the start of each sequence (called "[CLS]" for BERT). The
  dtype must match the dtype of the input tensors to the layer.
- **end_value**: int/str/list/tuple. The id(s) or token(s) that are to be
  placed at the end of the last input segment (called "[SEP]" for
  BERT). The dtype must match the dtype of the input tensors to the
  layer.
- **sep_value**: int/str/list/tuple. The id(s) or token(s) that are to be
  placed at the end of every segment, except the last segment (called
  "[SEP]" for BERT). If `None`, `end_value` is used. The dtype must
  match the dtype of the input tensors to the layer.
- **pad_value**: int/str. The id or token that is to be placed into the unused
  positions after the last segment in the sequence
  (called "[PAD]" for BERT).
- **truncate**: str. The algorithm to truncate a list of batched segments to
  fit a per-example length limit. The value can be either
  `"round_robin"` or `"waterfall"`:
  - `"round_robin"`: Available space is assigned one token at a
    time in a round-robin fashion to the inputs that still need
    some, until the limit is reached.
  - `"waterfall"`: The allocation of the budget is done using a
    "waterfall" algorithm that allocates quota in a
    left-to-right manner and fills up the buckets until we run
    out of budget. It support arbitrary number of segments.

**Returns**

A tuple with two elements. The first is the dense, packed token
sequence. The second is an integer tensor of the same shape, containing
the segment ids.

**Examples**

_Pack a single input for classification._

```console
>>> seq1 = [1, 2, 3, 4]
>>> packer = keras_hub.layers.MultiSegmentPacker(
...     sequence_length=8, start_value=101, end_value=102
... )
>>> token_ids, segment_ids = packer((seq1,))
>>> np.array(token_ids)
array([101, 1, 2, 3, 4, 102, 0, 0], dtype=int32)
>>> np.array(segment_ids)
array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
```

_Pack multiple inputs for classification._

```console
>>> seq1 = [1, 2, 3, 4]
>>> seq2 = [11, 12, 13, 14]
>>> packer = keras_hub.layers.MultiSegmentPacker(
...     sequence_length=8, start_value=101, end_value=102
... )
>>> token_ids, segment_ids = packer((seq1, seq2))
>>> np.array(token_ids)
array([101, 1, 2, 3, 102,  11,  12, 102], dtype=int32)
>>> np.array(segment_ids)
array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32)
```

_Pack multiple inputs for classification with different sep tokens._

```console
>>> seq1 = [1, 2, 3, 4]
>>> seq2 = [11, 12, 13, 14]
>>> packer = keras_hub.layers.MultiSegmentPacker(
...     sequence_length=8,
...     start_value=101,
...     end_value=102,
...     sep_value=[102, 102],
... )
>>> token_ids, segment_ids = packer((seq1, seq2))
>>> np.array(token_ids)
array([101,   1,   2, 102, 102,  11,  12, 102], dtype=int32)
>>> np.array(segment_ids)
array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32)
```

**Reference**

[Devlin et al., 2018](https://arxiv.org/abs/1810.04805).
