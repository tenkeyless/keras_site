---
title: EinsumDense layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/core/einsum_dense.py#L19" >}}

### `EinsumDense` class

```python
keras.layers.EinsumDense(
    equation,
    output_shape,
    activation=None,
    bias_axes=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    **kwargs
)
```

A layer that uses `einsum` as the backing computation.

This layer can perform einsum calculations of arbitrary dimensionality.

**Arguments**

- **equation**: An equation describing the einsum to perform. This equation must be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis expression sequence.
- **output_shape**: The expected shape of the output tensor (excluding the batch dimension and any dimensions represented by ellipses). You can specify `None` for any dimension that is unknown or can be inferred from the input shape.
- **activation**: Activation function to use. If you don't specify anything, no activation is applied (that is, a "linear" activation: `a(x) = x`).
- **bias_axes**: A string containing the output dimension(s) to apply a bias to. Each character in the `bias_axes` string should correspond to a character in the output portion of the `equation` string.
- **kernel_initializer**: Initializer for the `kernel` weights matrix.
- **bias_initializer**: Initializer for the bias vector.
- **kernel_regularizer**: Regularizer function applied to the `kernel` weights matrix.
- **bias_regularizer**: Regularizer function applied to the bias vector.
- **kernel_constraint**: Constraint function applied to the `kernel` weights matrix.
- **bias_constraint**: Constraint function applied to the bias vector.
- **lora_rank**: Optional integer. If set, the layer's forward pass will implement LoRA (Low-Rank Adaptation) with the provided rank. LoRA sets the layer's kernel to non-trainable and replaces it with a delta over the original kernel, obtained via multiplying two lower-rank trainable matrices (the factorization happens on the last dimension). This can be useful to reduce the computation cost of fine-tuning large dense layers. You can also enable LoRA on an existing `EinsumDense` layer by calling `layer.enable_lora(rank)`.
- **\*\*kwargs**: Base layer keyword arguments, such as `name` and `dtype`.

**Examples**

**Biased dense layer with einsums**

This example shows how to instantiate a standard Keras dense layer using einsum operations. This example is equivalent to `keras.layers.Dense(64, use_bias=True)`.

```console
>>> layer = keras.layers.EinsumDense("ab,bc->ac",
...                                       output_shape=64,
...                                       bias_axes="c")
>>> input_tensor = keras.Input(shape=[32])
>>> output_tensor = layer(input_tensor)
>>> output_tensor.shape
(None, 64)
```

**Applying a dense layer to a sequence**

This example shows how to instantiate a layer that applies the same dense operation to every element in a sequence. Here, the `output_shape` has two values (since there are two non-batch dimensions in the output); the first dimension in the `output_shape` is `None`, because the sequence dimension `b` has an unknown shape.

```console
>>> layer = keras.layers.EinsumDense("abc,cd->abd",
...                                       output_shape=(None, 64),
...                                       bias_axes="d")
>>> input_tensor = keras.Input(shape=[32, 128])
>>> output_tensor = layer(input_tensor)
>>> output_tensor.shape
(None, 32, 64)
```

**Applying a dense layer to a sequence using ellipses**

This example shows how to instantiate a layer that applies the same dense operation to every element in a sequence, but uses the ellipsis notation instead of specifying the batch and sequence dimensions.

Because we are using ellipsis notation and have specified only one axis, the `output_shape` arg is a single value. When instantiated in this way, the layer can handle any number of sequence dimensions - including the case where no sequence dimension exists.

```console
>>> layer = keras.layers.EinsumDense("...x,xy->...y",
...                                       output_shape=64,
...                                       bias_axes="y")
>>> input_tensor = keras.Input(shape=[32, 128])
>>> output_tensor = layer(input_tensor)
>>> output_tensor.shape
(None, 32, 64)
```
