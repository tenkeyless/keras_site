---
title: transformer_encoder
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/transformer_encoder.py#L11" >}}

### `TransformerEncoder` class

`keras_hub.layers.TransformerEncoder(     intermediate_dim,     num_heads,     dropout=0,     activation="relu",     layer_norm_epsilon=1e-05,     kernel_initializer="glorot_uniform",     bias_initializer="zeros",     normalize_first=False,     **kwargs )`

Transformer encoder.

This class follows the architecture of the transformer encoder layer in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users can instantiate multiple instances of this class to stack up an encoder.

This layer will correctly compute an attention mask from an implicit Keras padding mask (for example, by passing `mask_zero=True` to a [`keras.layers.Embedding`](/api/layers/core_layers/embedding#embedding-class) layer). See the Masking and Padding [guide](https://keras.io/guides/understanding_masking_and_padding/) for more details.

**Arguments**

- **intermediate_dim**: int, the hidden size of feedforward network.
- **num_heads**: int, the number of heads in the [`keras.layers.MultiHeadAttention`](/api/layers/attention_layers/multi_head_attention#multiheadattention-class) layer.
- **dropout**: float. the dropout value, shared by [`keras.layers.MultiHeadAttention`](/api/layers/attention_layers/multi_head_attention#multiheadattention-class) and feedforward network. Defaults to `0.`.
- **activation**: string or `keras.activations`. the activation function of feedforward network. Defaults to `"relu"`.
- **layer_norm_epsilon**: float. The epsilon value in layer normalization components. Defaults to `1e-5`.
- **kernel_initializer**: string or `keras.initializers` initializer. The kernel initializer for the dense and multiheaded attention layers. Defaults to `"glorot_uniform"`.
- **bias_initializer**: string or `keras.initializers` initializer. The bias initializer for the dense and multiheaded attention layers. Defaults to `"zeros"`.
- **normalize_first**: bool. If True, the inputs to the attention layer and the intermediate dense layer are normalized (similar to GPT-2). If set to False, outputs of attention layer and intermediate dense layer are normalized (similar to BERT). Defaults to `False`.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`](/api/layers/base_layer#layer-class), including `name`, `trainable`, `dtype` etc.

**Example**

`# Create a single transformer encoder layer. encoder = keras_hub.layers.TransformerEncoder(     intermediate_dim=64, num_heads=8)  # Create a simple model containing the encoder. input = keras.Input(shape=(10, 64)) output = encoder(input) model = keras.Model(inputs=input, outputs=output)  # Call encoder on the inputs. input_data = np.random.uniform(size=(2, 10, 64)) output = model(input_data)`

**References**

- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/transformer_encoder.py#L172" >}}

### `call` method

`TransformerEncoder.call(     inputs,     padding_mask=None,     attention_mask=None,     training=None,     return_attention_scores=False, )`

Forward pass of the TransformerEncoder.

**Arguments**

- **inputs**: a Tensor. The input data to TransformerEncoder, should be of shape \[batch_size, sequence_length, hidden_dim\].
- **padding_mask**: a boolean Tensor. It indicates if the token should be masked because the token is introduced due to padding. `padding_mask` should have shape \[batch_size, sequence_length\].
- **attention_mask**: a boolean Tensor. Customized mask used to mask out certain tokens. `attention_mask` should have shape \[batch_size, sequence_length, sequence_length\].
- **training**: a boolean indicating whether the layer should behave in training mode or in inference mode.
- **return_attention_scores**: a boolean indicating whether the output should be `(attention_output, attention_scores)` if `True` or `attention_output` if `False`. Defaults to `False`.

**Returns**

A Tensor of the same shape as the `inputs`.

---
