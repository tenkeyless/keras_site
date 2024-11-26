---
title: CachedMultiHeadAttention layer
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/cached_multi_head_attention.py#L7" >}}

### `CachedMultiHeadAttention` class

```python
keras_nlp.layers.CachedMultiHeadAttention(
    num_heads,
    key_dim,
    value_dim=None,
    dropout=0.0,
    use_bias=True,
    output_shape=None,
    attention_axes=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    seed=None,
    **kwargs
)
```

MultiHeadAttention layer with cache support.

This layer is suitable for use in autoregressive decoding. It can be used
to cache decoder self-attention and cross-attention. The forward pass
can happen in one of three modes:

- No cache, same as regular multi-head attention.
- Static cache (`cache_update_index` is None). In this case, the
  cached key/value projections will be used and the input values will
  be ignored.
- Updated cache (`cache_update_index` is not None). In this case, new
  key/value projections are computed using the input, and spliced into
  the cache at the specified index.

Note that caching is useful only during inference and should not be used
during training.

We use the notation `B`, `T`, `S` below, where `B` is the batch dimension,
`T` is the target sequence length, and `S` in the source sequence length.
Note that during generative decoding, `T` is usually 1 (you are
generating a target sequence of length one to predict the next token).

**Call arguments**

- **query**: Query `Tensor` of shape `(B, T, dim)`.
- **value**: Value `Tensor` of shape `(B, S*, dim)`. if `cache` is None`,`S\*`must equal`S`and match the shape of`attention_mask`. If cache` is
  not `None`, `S*` can be any length less than `S`, and the computed
  value will be spliced into `cache` at `cache_update_index`.
- **key**: Optional key `Tensor` of shape `(B, S*, dim)`. If `cache` is
  `None`, `S*` must equal `S` and match the shape of
  `attention_mask`. If `cache` is not `None`, `S*` can be any length
  less than `S`, and the computed value will be spliced into `cache`
  at `cache_update_index`.
- **attention_mask**: a boolean mask of shape `(B, T, S)`. `attention_mask`
  prevents attention to certain positions. The boolean mask specifies
  which query elements can attend to which key elements, 1 indicates
  attention and 0 indicates no attention. Broadcasting can happen for
  the missing batch dimensions and the head dimension.
- **cache**: a dense float Tensor. The key/value cache, of shape
  `[B, 2, S, num_heads, key_dims]`, where `S` must agree with the
  `attention_mask` shape. This argument is intended for use during
  generation to avoid recomputing intermediate state.
- **cache_update_index**: a int or int Tensor, the index at which to update
  `cache` (usually the index of the current token being processed
  when running generation). If `cache_update_index=None` while `cache`
  is set, the cache will not be updated.
- **training**: a boolean indicating whether the layer should behave in
  training mode or in inference mode.

**Returns**

An `(attention_output, cache)` tuple. `attention_output` is the result
of the computation, of shape `(B, T, dim)`, where `T` is for target
sequence shapes and `dim` is the query input last dimension if
`output_shape` is `None`. Otherwise, the multi-head outputs are
projected to the shape specified by `output_shape`. `cache` is the
updated cache.
