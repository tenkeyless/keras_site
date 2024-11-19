---
title: masked_lm_head
toc: false
---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/masked_lm_head.py#L7)

### `MaskedLMHead` class

`keras_nlp.layers.MaskedLMHead(     vocabulary_size=None,     token_embedding=None,     intermediate_activation="relu",     activation=None,     layer_norm_epsilon=1e-05,     kernel_initializer="glorot_uniform",     bias_initializer="zeros",     **kwargs )`

Masked Language Model (MaskedLM) head.

This layer takes two inputs:

- `inputs`: which should be a tensor of encoded tokens with shape `(batch_size, sequence_length, hidden_dim)`.
- `mask_positions`: which should be a tensor of integer positions to predict with shape `(batch_size, masks_per_sequence)`.

The token encodings should usually be the last output of an encoder model, and mask positions should be the integer positions you would like to predict for the MaskedLM task.

The layer will first gather the token encodings at the mask positions. These gathered tokens will be passed through a dense layer the same size as encoding dimension, then transformed to predictions the same size as the input vocabulary. This layer will produce a single output with shape `(batch_size, masks_per_sequence, vocabulary_size)`, which can be used to compute an MaskedLM loss function.

This layer is often be paired with [`keras_hub.layers.MaskedLMMaskGenerator`](/api/keras_hub/preprocessing_layers/masked_lm_mask_generator#maskedlmmaskgenerator-class), which will help prepare inputs for the MaskedLM task.

**Arguments**

- **vocabulary_size**: The total size of the vocabulary for predictions.
- **token_embedding**: Optional. A [`keras_hub.layers.ReversibleEmbedding`](/api/keras_hub/modeling_layers/reversible_embedding#reversibleembedding-class) instance. If passed, the layer will be used to project from the `hidden_dim` of the model to the output `vocabulary_size`.
- **intermediate_activation**: The activation function of intermediate dense layer.
- **activation**: The activation function for the outputs of the layer. Usually either `None` (return logits), or `"softmax"` (return probabilities).
- **layer_norm_epsilon**: float. The epsilon value in layer normalization components. Defaults to `1e-5`.
- **kernel_initializer**: string or `keras.initializers` initializer. The kernel initializer for the dense and multiheaded attention layers. Defaults to `"glorot_uniform"`.
- **bias_initializer**: string or `keras.initializers` initializer. The bias initializer for the dense and multiheaded attention layers. Defaults to `"zeros"`.
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`](/api/layers/base_layer#layer-class), including `name`, `trainable`, `dtype` etc.

**Example**

`` batch_size = 16 vocab_size = 100 hidden_dim = 32 seq_length = 50  # Generate random inputs. token_ids = np.random.randint(vocab_size, size=(batch_size, seq_length)) # Choose random positions as the masked inputs. mask_positions = np.random.randint(seq_length, size=(batch_size, 5))  # Embed tokens in a `hidden_dim` feature space. token_embedding = keras_hub.layers.ReversibleEmbedding(     vocab_size,     hidden_dim, ) hidden_states = token_embedding(token_ids)  preds = keras_hub.layers.MaskedLMHead(     vocabulary_size=vocab_size,     token_embedding=token_embedding,     activation="softmax", )(hidden_states, mask_positions) ``

**References**

- [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)

---
