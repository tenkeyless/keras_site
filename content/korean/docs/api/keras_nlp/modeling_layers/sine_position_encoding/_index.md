---
title: SinePositionEncoding layer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/sine_position_encoding.py#L7" >}}

### `SinePositionEncoding` class

```python
keras_nlp.layers.SinePositionEncoding(max_wavelength=10000, **kwargs)
```

Sinusoidal positional encoding layer.

This layer calculates the position encoding as a mix of sine and cosine
functions with geometrically increasing wavelengths. Defined and formulized
in [Attention is All You Need](https://arxiv.org/abs/1706.03762).

Takes as input an embedded token tensor. The input must have shape
[batch\_size, sequence\_length, feature\_size]. This layer will return a
positional encoding the same size as the embedded token tensor, which
can be added directly to the embedded token tensor.

**Arguments**

- **max_wavelength**: The maximum angular wavelength of the sine/cosine
  curves, as described in Attention is All You Need. Defaults to
  `10000`.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}),
  including `name`, `trainable`, `dtype` etc.

**Call arguments**

- **inputs**: The tensor inputs to compute an embedding for, with shape
  `(batch_size, sequence_length, hidden_dim)`.
- **start_index**: An integer or integer tensor. The starting position to
  compute the encoding from. This is useful during cached decoding,
  where each position is predicted separately in a loop.

**Example**

```python
# create a simple embedding layer with sinusoidal positional encoding
seq_len = 100
vocab_size = 1000
embedding_dim = 32
inputs = keras.Input((seq_len,), dtype="float32")
embedding = keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embedding_dim
)(inputs)
positional_encoding = keras_hub.layers.SinePositionEncoding()(embedding)
outputs = embedding + positional_encoding
```

**References**

- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
