---
title: TransformerDecoder layer
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/transformer_decoder.py#L16" >}}

### `TransformerDecoder` class

`keras_nlp.layers.TransformerDecoder(     intermediate_dim,     num_heads,     dropout=0,     activation="relu",     layer_norm_epsilon=1e-05,     kernel_initializer="glorot_uniform",     bias_initializer="zeros",     normalize_first=False,     **kwargs )`

Transformer decoder.

This class follows the architecture of the transformer decoder layer in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users can instantiate multiple instances of this class to stack up a decoder.

By default, this layer will apply a causal mask to the decoder attention layer. You can also pass padding or attention masks directly to the layer during call, e.g. with `decoder_padding_mask` or `decoder_attention_mask`.

This layer can be called with either one or two inputs. The number of inputs must be consistent across all calls. The options are as follows: `layer(decoder_sequence)`: no cross-attention will be built into the decoder block. This is useful when building a "decoder-only" transformer such as GPT-2. `layer(decoder_sequence, encoder_sequence)`: cross-attention will be built into the decoder block. This is useful when building an "encoder-decoder" transformer, such as the original transformer model described in Attention is All You Need.

**Arguments**

- **intermediate_dim**: int, the hidden size of feedforward network.
- **num_heads**: int, the number of heads in MultiHeadAttention.
- **dropout**: float. the dropout value, shared by MultiHeadAttention and feedforward network. Defaults to `0.`.
- **activation**: string or `keras.activations`. the activation function of feedforward network. Defaults to `"relu"`.
- **layer_norm_epsilon**: float. The eps value in layer normalization components. Defaults to `1e-5`.
- **kernel_initializer**: string or `keras.initializers` initializer. The kernel initializer for the dense and multiheaded attention layers. Defaults to `"glorot_uniform"`.
- **bias_initializer**: string or `keras.initializers` initializer. The bias initializer for the dense and multiheaded attention layers. Defaults to `"zeros"`.
- **normalize_first**: bool. If True, the inputs to the attention layer(s) and the intermediate dense layer are normalized (similar to GPT-2). If set to False, outputs of attention layer and intermediate dense layer are normalized (similar to BERT). Defaults to `False`.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`](/api/layers/base_layer#layer-class), including `name`, `trainable`, `dtype` etc.

**Example**

`# Create a single transformer decoder layer. decoder = keras_hub.layers.TransformerDecoder(     intermediate_dim=64, num_heads=8)  # Create a simple model containing the decoder. decoder_input = keras.Input(shape=(10, 64)) encoder_input = keras.Input(shape=(10, 64)) output = decoder(decoder_input, encoder_input) model = keras.Model(     inputs=(decoder_input, encoder_input),     outputs=output, )  # Call decoder on the inputs. decoder_input_data = np.random.uniform(size=(2, 10, 64)) encoder_input_data = np.random.uniform(size=(2, 10, 64)) decoder_output = model((decoder_input_data, encoder_input_data))`

**References**

- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/transformer_decoder.py#L240" >}}

### `call` method

`TransformerDecoder.call(     decoder_sequence,     encoder_sequence=None,     decoder_padding_mask=None,     decoder_attention_mask=None,     encoder_padding_mask=None,     encoder_attention_mask=None,     self_attention_cache=None,     self_attention_cache_update_index=None,     cross_attention_cache=None,     cross_attention_cache_update_index=None,     use_causal_mask=True,     training=None, )`

Forward pass of the TransformerDecoder.

**Arguments**

- **decoder_sequence**: a Tensor. The decoder input sequence.
- **encoder_sequence**: a Tensor. The encoder input sequence. For decoder only models (like GPT2), this should be left `None`. Once the model is called once without an encoder_sequence, you cannot call it again with encoder_sequence.
- **decoder_padding_mask**: a boolean Tensor, the padding mask of decoder sequence, must be of shape `[batch_size, decoder_sequence_length]`.
- **decoder_attention_mask**: a boolean Tensor. Customized decoder sequence mask, must be of shape `[batch_size, decoder_sequence_length, decoder_sequence_length]`.
- **encoder_padding_mask**: a boolean Tensor, the padding mask of encoder sequence, must be of shape `[batch_size, encoder_sequence_length]`.
- **encoder_attention_mask**: a boolean Tensor. Customized encoder sequence mask, must be of shape `[batch_size, encoder_sequence_length, encoder_sequence_length]`.
- **self_attention_cache**: a dense float Tensor. The cache of key/values pairs in the self-attention layer. Has shape `[batch_size, 2, max_seq_len, num_heads, key_dims]`.
- **self_attention_cache_update_index**: an int or int Tensor, the index at which to update the `self_attention_cache`. Usually, this is the index of the current token being processed during decoding.
- **cross_attention_cache**: a dense float Tensor. The cache of key/value pairs in the cross-attention layer. Has shape `[batch_size, 2, S, num_heads, key_dims]`.
- **cross_attention_cache_update_index**: an int or int Tensor, the index at which to update the `cross_attention_cache`. Usually, this is either `0` (compute the entire `cross_attention_cache`), or `None` (reuse a previously computed `cross_attention_cache`).
- **use_causal_mask**: bool, defaults to `True`. If true, a causal mask (masking out future input) is applied \`on the decoder sequence.
- **training**: a boolean indicating whether the layer should behave in training mode or in inference mode.

**Returns**

- **One of three things, depending on call arguments**:
- `outputs`, if `self_attention_cache` is \`None.
- `(outputs, self_attention_cache)`, if `self_attention_cache` is set and the layer has no cross-attention.
- `(outputs, self_attention_cache, cross_attention_cache)`, if `self_attention_cache` and `cross_attention_cache` are set and the layer has cross-attention.

---
