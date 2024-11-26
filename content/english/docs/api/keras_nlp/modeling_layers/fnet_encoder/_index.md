---
title: FNetEncoder layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/f_net_encoder.py#L8" >}}

### `FNetEncoder` class

```python
keras_nlp.layers.FNetEncoder(
    intermediate_dim,
    dropout=0,
    activation="relu",
    layer_norm_epsilon=1e-05,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    **kwargs
)
```

FNet encoder.

This class follows the architecture of FNet encoder layer in the
[FNet paper](https://arxiv.org/abs/2105.03824). Users can instantiate
multiple instances of this class to stack up the encoder.

Note on masking: In the official FNet code, padding tokens are added to the
the input. However, the padding masks are deleted, i.e., mixing of
all tokens is done. This is because certain frequencies will be zeroed
out if we apply padding masks in every encoder layer. Hence, we don't
take padding mask as input in the call() function.

**Arguments**

- **intermediate_dim**: int. The hidden size of feedforward network.
- **dropout**: float. The dropout value, applied in the
  feedforward network. Defaults to `0.`.
- **activation**: string or `keras.activations`. The
  activation function of feedforward network.
  Defaults to `"relu"`.
- **layer_norm_epsilon**: float. The epsilon value in layer
  normalization components. Defaults to `1e-5`.
- **kernel_initializer**: `str` or `keras.initializers` initializer.
  The kernel initializer for the dense layers.
  Defaults to `"glorot_uniform"`.
- **bias_initializer**: "string" or `keras.initializers` initializer.
  The bias initializer for the dense layers.
  Defaults to `"zeros"`.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}),
  including `name`, `trainable`, `dtype` etc.

**Example**

```python
# Create a single FNet encoder layer.
encoder = keras_hub.layers.FNetEncoder(
    intermediate_dim=64)
# Create a simple model containing the encoder.
input = keras.Input(shape=(10, 64))
output = encoder(input)
model = keras.Model(inputs=input, outputs=output)
# Call encoder on the inputs.
input_data = np.random.uniform(size=(1, 10, 64))
output = model(input_data)
```

**References**

- [Lee-Thorp et al., 2021](https://arxiv.org/abs/2105.03824)
