---
title: PositionEmbedding layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/position_embedding.py#L7" >}}

### `PositionEmbedding` class

```python
keras_hub.layers.PositionEmbedding(
    sequence_length, initializer="glorot_uniform", **kwargs
)
```

A layer which learns a position embedding for inputs sequences.

This class assumes that in the input tensor, the last dimension corresponds
to the features, and the dimension before the last corresponds to the
sequence.

This layer does not supporting masking, but can be combined with a
[`keras.layers.Embedding`]({{< relref "/docs/api/layers/core_layers/embedding#embedding-class" >}}) for padding mask support.

**Arguments**

- **sequence_length**: The maximum length of the dynamic sequence.
- **initializer**: The initializer to use for the embedding weights. Defaults
  to `"glorot_uniform"`.
- **seq_axis**: The axis of the input tensor where we add the embeddings.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}),
  including `name`, `trainable`, `dtype` etc.

**Call arguments**

- **inputs**: The tensor inputs to compute an embedding for, with shape
  `(batch_size, sequence_length, hidden_dim)`. Only the input shape
  will be used, as the position embedding does not depend on the
  input sequence content.
- **start_index**: An integer or integer tensor. The starting position to
  compute the position embedding from. This is useful during cached
  decoding, where each position is predicted separately in a loop.

**Example**

Called directly on input.

```console
>>> layer = keras_hub.layers.PositionEmbedding(sequence_length=10)
>>> layer(np.zeros((8, 10, 16)))
```

Combine with a token embedding.

```python
seq_length = 50
vocab_size = 5000
embed_dim = 128
inputs = keras.Input(shape=(seq_length,))
token_embeddings = keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embed_dim
)(inputs)
position_embeddings = keras_hub.layers.PositionEmbedding(
    sequence_length=seq_length
)(token_embeddings)
outputs = token_embeddings + position_embeddings
```

**Reference**

- [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
