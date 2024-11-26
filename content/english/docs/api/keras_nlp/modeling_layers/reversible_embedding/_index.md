---
title: ReversibleEmbedding layer
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/reversible_embedding.py#L9" >}}

### `ReversibleEmbedding` class

```python
keras_nlp.layers.ReversibleEmbedding(
    input_dim,
    output_dim,
    tie_weights=True,
    embeddings_initializer="uniform",
    embeddings_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    reverse_dtype=None,
    logit_soft_cap=None,
    **kwargs
)
```

An embedding layer which can project backwards to the input dim.

This layer is an extension of [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) for language models.
This layer can be called "in reverse" with `reverse=True`, in which case the
layer will linearly project from `output_dim` back to `input_dim`.

By default, the reverse projection will use the transpose of the
`embeddings` weights to project to `input_dim` (weights are "tied"). If
`tie_weights=False`, the model will use a separate, trainable variable for
reverse projection.

This layer has no bias terms.

**Arguments**

- **input_dim**: Integer. Size of the vocabulary,
  i.e. maximum integer index + 1.
- **output_dim**: Integer. Dimension of the dense embedding.
- **tie_weights**: Boolean, whether or not the matrix for embedding and
  the matrix for the `reverse` projection should share the same
  weights.
- **embeddings_initializer**: Initializer for the `embeddings`
  matrix (see `keras.initializers`).
- **embeddings_regularizer**: Regularizer function applied to
  the `embeddings` matrix (see `keras.regularizers`).
- **embeddings_constraint**: Constraint function applied to
  the `embeddings` matrix (see `keras.constraints`).
- **mask_zero**: Boolean, whether or not the input value 0 is a special
  "padding" value that should be masked out.
- **reverse_dtype**: The dtype for the reverse projection computation.
  Defaults to the `compute_dtype` of the layer.
- **logit_soft_cap**: If `logit_soft_cap` is set and `reverse=True`, the
  output logits will be scaled by
  `tanh(logits / logit_soft_cap) * logit_soft_cap`. This narrows the
  range of output logits and can improve training.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}),
  including `name`, `trainable`, `dtype` etc.

**Call arguments**

- **inputs**: The tensor inputs to the layer.
- **reverse**: Boolean. If `True` the layer will perform a linear projection
  from `output_dim` to `input_dim`, instead of a normal embedding
  call. Default to `False`.

**Example**

```python
batch_size = 16
vocab_size = 100
hidden_dim = 32
seq_length = 50
# Generate random inputs.
token_ids = np.random.randint(vocab_size, size=(batch_size, seq_length))
embedding = keras_hub.layers.ReversibleEmbedding(vocab_size, hidden_dim)
# Embed tokens to shape `(batch_size, seq_length, hidden_dim)`.
hidden_states = embedding(token_ids)
# Project hidden states to shape `(batch_size, seq_length, vocab_size)`.
logits = embedding(hidden_states, reverse=True)
```

**References**

- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
