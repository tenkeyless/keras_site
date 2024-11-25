---
title: Attention layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/attention/attention.py#L7" >}}

### `Attention` class

```python
keras.layers.Attention(
    use_scale=False, score_mode="dot", dropout=0.0, seed=None, **kwargs
)
```

Dot-product attention layer, a.k.a. Luong-style attention.

Inputs are a list with 2 or 3 elements: 1. A `query` tensor of shape `(batch_size, Tq, dim)`. 2. A `value` tensor of shape `(batch_size, Tv, dim)`. 3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none supplied, `value` will be used as a `key`.

The calculation follows the steps: 1. Calculate attention scores using `query` and `key` with shape `(batch_size, Tq, Tv)`. 2. Use scores to calculate a softmax distribution with shape `(batch_size, Tq, Tv)`. 3. Use the softmax distribution to create a linear combination of `value` with shape `(batch_size, Tq, dim)`.

**Arguments**

- **use_scale**: If `True`, will create a scalar variable to scale the attention scores.
- **dropout**: Float between 0 and 1. Fraction of the units to drop for the attention scores. Defaults to `0.0`.
- **seed**: A Python integer to use as random seed incase of `dropout`.
- **score_mode**: Function to use to compute attention scores, one of `{"dot", "concat"}`. `"dot"` refers to the dot product between the query and key vectors. `"concat"` refers to the hyperbolic tangent of the concatenation of the `query` and `key` vectors.

**Call arguments**

- **inputs**: List of the following tensors:
  - `query`: Query tensor of shape `(batch_size, Tq, dim)`.
  - `value`: Value tensor of shape `(batch_size, Tv, dim)`.
  - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If not given, will use `value` for both `key` and `value`, which is the most common case.
- **mask**: List of the following tensors:
  - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`. If given, the output will be zero at the positions where `mask==False`.
  - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`. If given, will apply the mask such that values at positions where `mask==False` do not contribute to the result.
- **return_attention_scores**: bool, it `True`, returns the attention scores (after masking and softmax) as an additional output argument.
- **training**: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout).
- **use_causal_mask**: Boolean. Set to `True` for decoder self-attention. Adds a mask such that position `i` cannot attend to positions `j > i`. This prevents the flow of information from the future towards the past. Defaults to `False`.

Output: Attention outputs of shape `(batch_size, Tq, dim)`. (Optional) Attention scores after masking and softmax with shape `(batch_size, Tq, Tv)`.
