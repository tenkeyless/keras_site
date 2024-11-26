---
title: RotaryEmbedding layer
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/rotary_embedding.py#L7" >}}

### `RotaryEmbedding` class

```python
keras_nlp.layers.RotaryEmbedding(
    max_wavelength=10000, scaling_factor=1.0, sequence_axis=1, feature_axis=-1, **kwargs
)
```

Rotary positional encoding layer.

This layer encodes absolute positional information with a rotation
matrix. It calculates the rotary encoding with a mix of sine and
cosine functions with geometrically increasing wavelengths.
Defined and formulated in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4).
The input must be a tensor with shape a sequence dimension and a feature
dimension. Typically, this will either an input with shape
`(batch_size, sequence_length, feature_length)` or
`(batch_size, sequence_length, num_heads, feature_length)`.
This layer will return a new tensor with the rotary embedding applied to
the input tensor.

**Arguments**

- **max_wavelength**: int. The maximum angular wavelength of the sine/cosine
  curves.
- **scaling_factor**: float. The scaling factor used to scale positions of
  the tokens.
- **sequence_axis**: int. Sequence axis in the input tensor.
- **feature_axis**: int. Feature axis in the input tensor.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}),
  including `name`, `trainable`, `dtype` etc.

**Call arguments**

- **inputs**: The tensor inputs to apply the embedding to. This can have
  any shape, but must contain both a sequence and feature axis. The
  rotary embedding will be applied to `inputs` and returned.
- **start_index**: An integer or integer tensor. The starting position to
  compute the rotary embedding from. This is useful during cached
  decoding, where each position is predicted separately in a loop.

**Examples**

```python
batch_size = 16
feature_length = 18
sequence_length = 256
num_heads = 8
# No multi-head dimension.
tensor = np.ones((batch_size, sequence_length, feature_length))
rot_emb_layer = RotaryEmbedding()
tensor_rot = rot_emb_layer(tensor)
# With multi-head dimension.
tensor = np.ones((batch_size, sequence_length, num_heads, feature_length))
tensor_rot = rot_emb_layer(tensor)
```

**References**

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4)
