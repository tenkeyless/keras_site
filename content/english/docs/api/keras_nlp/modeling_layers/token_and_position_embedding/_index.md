---
title: token_and_position_embedding
toc: false
---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/modeling/token_and_position_embedding.py#L11)

### `TokenAndPositionEmbedding` class

`keras_nlp.layers.TokenAndPositionEmbedding(     vocabulary_size,     sequence_length,     embedding_dim,     tie_weights=True,     embeddings_initializer="uniform",     mask_zero=False,     **kwargs )`

A layer which sums a token and position embedding.

Token and position embeddings are ways of representing words and their order in a sentence. This layer creates a [`keras.layers.Embedding`](/api/layers/core_layers/embedding#embedding-class) token embedding and a [`keras_hub.layers.PositionEmbedding`](/api/keras_hub/modeling_layers/position_embedding#positionembedding-class) position embedding and sums their output when called. This layer assumes that the last dimension in the input corresponds to the sequence dimension.

**Arguments**

- **vocabulary_size**: The size of the vocabulary.
- **sequence_length**: The maximum length of input sequence
- **embedding_dim**: The output dimension of the embedding layer
- **tie_weights**: Boolean, whether or not the matrix for embedding and the matrix for the `reverse` projection should share the same weights.
- **embeddings_initializer**: The initializer to use for the Embedding Layers
- **mask_zero**: Boolean, whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If this is True, then all subsequent layers in the model need to support masking or an exception will be raised. If mask_zero\` is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
- **\*\*kwargs**: other keyword arguments passed to [`keras.layers.Layer`](/api/layers/base_layer#layer-class), including `name`, `trainable`, `dtype` etc.

**Example**

`inputs = np.ones(shape=(1, 50), dtype="int32") embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(     vocabulary_size=10_000,     sequence_length=50,     embedding_dim=128, ) outputs = embedding_layer(inputs)`

---
