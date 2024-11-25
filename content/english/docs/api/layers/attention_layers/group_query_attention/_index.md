---
title: GroupQueryAttention
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/attention/grouped_query_attention.py#L12" >}}

### `GroupedQueryAttention` class

```python
keras.layers.GroupQueryAttention(
    head_dim,
    num_query_heads,
    num_key_value_heads,
    dropout=0.0,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

Grouped Query Attention layer.

This is an implementation of grouped-query attention introduced by [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245). Here `num_key_value_heads` denotes number of groups, setting `num_key_value_heads` to 1 is equivalent to multi-query attention, and when `num_key_value_heads` is equal to `num_query_heads` it is equivalent to multi-head attention.

This layer first projects `query`, `key`, and `value` tensors. Then, `key` and `value` are repeated to match the number of heads of `query`.

Then, the `query` is scaled and dot-producted with `key` tensors. These are softmaxed to obtain attention probabilities. The value tensors are then interpolated by these probabilities and concatenated back to a single tensor.

**Arguments**

- **head_dim**: Size of each attention head.
- **num_query_heads**: Number of query attention heads.
- **num_key_value_heads**: Number of key and value attention heads.
- **dropout**: Dropout probability.
- **use_bias**: Boolean, whether the dense layers use bias vectors/matrices.
- **kernel_initializer**: Initializer for dense layer kernels.
- **bias_initializer**: Initializer for dense layer biases.
- **kernel_regularizer**: Regularizer for dense layer kernels.
- **bias_regularizer**: Regularizer for dense layer biases.
- **activity_regularizer**: Regularizer for dense layer activity.
- **kernel_constraint**: Constraint for dense layer kernels.
- **bias_constraint**: Constraint for dense layer kernels.

**Call arguments**

- **query**: Query tensor of shape `(batch_dim, target_seq_len, feature_dim)`, where `batch_dim` is batch size, `target_seq_len` is the length of target sequence, and `feature_dim` is dimension of feature.
- **value**: Value tensor of shape `(batch_dim, source_seq_len, feature_dim)`, where `batch_dim` is batch size, `source_seq_len` is the length of source sequence, and `feature_dim` is dimension of feature.
- **key**: Optional key tensor of shape `(batch_dim, source_seq_len, feature_dim)`. If not given, will use `value` for both `key` and `value`, which is most common case.
- **attention_mask**: A boolean mask of shape `(batch_dim, target_seq_len, source_seq_len)`, that prevents attention to certain positions. The boolean mask specifies which query elements can attend to which key elements, where 1 indicates attention and 0 indicates no attention. Broadcasting can happen for the missing batch dimensions and the head dimension.
- **return_attention_scores**: A boolean to indicate whether the output should be `(attention_output, attention_scores)` if `True`, or `attention_output` if `False`. Defaults to `False`.
- **training**: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout). Will go with either using the training mode of the parent layer/model or `False` (inference) if there is no parent layer.
- **use_causal_mask**: A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens (e.g., used in a decoder Transformer).

**Returns**

- **attention_output**: Result of the computation, of shape `(batch_dim, target_seq_len, feature_dim)`, where `target_seq_len` is for target sequence length and `feature_dim` is the query input last dim.
- **attention_scores**: (Optional) attention coefficients of shape `(batch_dim, num_query_heads, target_seq_len, source_seq_len)`.
