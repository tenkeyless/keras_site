---
title: AlibiBias layer
toc: true
weight: 9
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/alibi_bias.py#L9" >}}

### `AlibiBias` class

```python
keras_nlp.layers.AlibiBias(alibi_bias_max=8, **kwargs)
```

A layer that adds the alibi bias to attention scores.

This layer adds the alibi bias to the attention scores. Alibi bias is a
linear, non-learned bias. Defined and formalized in
[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).

This layer takes as input the attention scores. and returns the attention
scores after adding the alibi bias to it. The output will have the same
shape as the input.

**Arguments**

- **alibi_bias_max**: int. This value will be used to compute the slope of
  each head. The heads' slopes are a geometric sequence that starts at
  `2**(-alibi_bias_max/num_heads)` and uses that same value as its
  ratio. Defaults to 8.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}),
  including `name`, `trainable`, `dtype` etc.

**Call arguments**

- **attention_scores**: The result of multipying the query and the key of the
  multi-head attention layer of the transformer to add alibi bias to
  it. With shape `(batch_size, num_heads, query_length, key_length)`.

**Example**

```python
query_length = 10
key_length = 10
num_heads = 4
batch_size = 2
hidden_dim = 8
# Create new alibi layer.
alibi_layer = keras_hub.layers.AlibiBias()
query = np.zeros((batch_size, num_heads, query_length, hidden_dim))
key = np.zeros((batch_size, num_heads, hidden_dim, key_length))
attention_scores = keras.ops.matmul(query, key)
# Add alibi bias to attention scores.
attention_scores = alibi_layer(attention_scores)
```

**References**

- [Press et al., 2021](https://arxiv.org/abs/2108.12409)
