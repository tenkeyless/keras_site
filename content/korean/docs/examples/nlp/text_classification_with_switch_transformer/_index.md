---
title: 스위치 트랜스포머로 텍스트 분류
linkTitle: 스위치 트랜스포머 텍스트 분류
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-21" >}}

**{{< t f_author >}}** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)  
**{{< t f_date_created >}}** 2020/05/10  
**{{< t f_last_modified >}}** 2021/02/15  
**{{< t f_description >}}** Implement a Switch Transformer for text classification.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_classification_with_switch_transformer.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_switch_transformer.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Introduction {#introduction}

This example demonstrates the implementation of the [Switch Transformer](https://arxiv.org/abs/2101.03961) model for text classification.

The Switch Transformer replaces the feedforward network (FFN) layer in the standard Transformer with a Mixture of Expert (MoE) routing layer, where each expert operates independently on the tokens in the sequence. This allows increasing the model size without increasing the computation needed to process each example.

Note that, for training the Switch Transformer efficiently, data and model parallelism need to be applied, so that expert modules can run simultaneously, each on its own accelerator. While the implementation described in the paper uses the [TensorFlow Mesh](https://github.com/tensorflow/mesh) framework for distributed training, this example presents a simple, non-distributed implementation of the Switch Transformer model for demonstration purposes.

## Setup {#setup}

```python
import keras
from keras import ops
from keras import layers
```

## Download and prepare dataset {#download-and-prepare-dataset}

```python
vocab_size = 20000  # Only consider the top 20k words
num_tokens_per_example = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.utils.pad_sequences(x_train, maxlen=num_tokens_per_example)
x_val = keras.utils.pad_sequences(x_val, maxlen=num_tokens_per_example)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
25000 Training sequences
25000 Validation sequences
```

{{% /details %}}

## Define hyperparameters {#define-hyperparameters}

```python
embed_dim = 32  # Embedding size for each token.
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feedforward network.
num_experts = 10  # Number of experts used in the Switch Transformer.
batch_size = 50  # Batch size.
learning_rate = 0.001  # Learning rate.
dropout_rate = 0.25  # Dropout rate.
num_epochs = 3  # Number of epochs.
num_tokens_per_batch = (
    batch_size * num_tokens_per_example
)  # Total number of tokens per batch.
print(f"Number of tokens per batch: {num_tokens_per_batch}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Number of tokens per batch: 10000
```

{{% /details %}}

## Implement token & position embedding layer {#implement-token-position-embedding-layer}

It consists of two separate embedding layers, one for tokens, one for token index (positions).

```python
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
```

## Implement the feedforward network {#implement-the-feedforward-network}

This is used as the Mixture of Experts in the Switch Transformer.

```python
def create_feedforward_network(ff_dim, embed_dim, name=None):
    return keras.Sequential(
        [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)], name=name
    )
```

## Implement the load-balanced loss {#implement-the-load-balanced-loss}

This is an auxiliary loss to encourage a balanced load across experts.

```python
def load_balanced_loss(router_probs, expert_mask):
    # router_probs [tokens_per_batch, num_experts] is the probability assigned for
    # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
    # the expert with the highest router probability in one−hot format.

    num_experts = ops.shape(expert_mask)[-1]
    # Get the fraction of tokens routed to each expert.
    # density is a vector of length num experts that sums to 1.
    density = ops.mean(expert_mask, axis=0)
    # Get fraction of probability mass assigned to each expert from the router
    # across all tokens. density_proxy is a vector of length num experts that sums to 1.
    density_proxy = ops.mean(router_probs, axis=0)
    # Want both vectors to have uniform allocation (1/num experts) across all
    # num_expert elements. The two vectors will be pushed towards uniform allocation
    # when the dot product is minimized.
    loss = ops.mean(density_proxy * density) * ops.cast((num_experts**2), "float32")
    return loss
```

### Implement the router as a layer {#implement-the-router-as-a-layer}

```python
class Router(layers.Layer):
    def __init__(self, num_experts, expert_capacity):
        self.num_experts = num_experts
        self.route = layers.Dense(units=num_experts)
        self.expert_capacity = expert_capacity
        super().__init__()

    def call(self, inputs, training=False):
        # inputs shape: [tokens_per_batch, embed_dim]
        # router_logits shape: [tokens_per_batch, num_experts]
        router_logits = self.route(inputs)

        if training:
            # Add noise for exploration across experts.
            router_logits += keras.random.uniform(
                shape=router_logits.shape, minval=0.9, maxval=1.1
            )
        # Probabilities for each token of what expert it should be sent to.
        router_probs = keras.activations.softmax(router_logits, axis=-1)
        # Get the top−1 expert for each token. expert_gate is the top−1 probability
        # from the router for each token. expert_index is what expert each token
        # is going to be routed to.
        expert_gate, expert_index = ops.top_k(router_probs, k=1)
        # expert_mask shape: [tokens_per_batch, num_experts]
        expert_mask = ops.one_hot(expert_index, self.num_experts)
        # Compute load balancing loss.
        aux_loss = load_balanced_loss(router_probs, expert_mask)
        self.add_loss(aux_loss)
        # Experts have a fixed capacity, ensure we do not exceed it. Construct
        # the batch indices, to each expert, with position in expert make sure that
        # not more that expert capacity examples can be routed to each expert.
        position_in_expert = ops.cast(
            ops.cumsum(expert_mask, axis=0) * expert_mask, "int32"
        )
        # Keep only tokens that fit within expert capacity.
        expert_mask *= ops.cast(
            ops.less(ops.cast(position_in_expert, "int32"), self.expert_capacity),
            "float32",
        )
        expert_mask_flat = ops.sum(expert_mask, axis=-1)
        # Mask out the experts that have overflowed the expert capacity.
        expert_gate *= expert_mask_flat
        # Combine expert outputs and scaling with router probability.
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        combined_tensor = ops.expand_dims(
            expert_gate
            * expert_mask_flat
            * ops.squeeze(ops.one_hot(expert_index, self.num_experts), 1),
            -1,
        ) * ops.squeeze(ops.one_hot(position_in_expert, self.expert_capacity), 1)
        # Create binary dispatch_tensor [tokens_per_batch, num_experts, expert_capacity]
        # that is 1 if the token gets routed to the corresponding expert.
        dispatch_tensor = ops.cast(combined_tensor, "float32")

        return dispatch_tensor, combined_tensor
```

### Implement a Switch layer {#implement-a-switch-layer}

```python
class Switch(layers.Layer):
    def __init__(
        self, num_experts, embed_dim, ff_dim, num_tokens_per_batch, capacity_factor=1
    ):
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.experts = [
            create_feedforward_network(ff_dim, embed_dim) for _ in range(num_experts)
        ]

        self.expert_capacity = num_tokens_per_batch // self.num_experts
        self.router = Router(self.num_experts, self.expert_capacity)
        super().__init__()

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        num_tokens_per_example = ops.shape(inputs)[1]

        # inputs shape: [num_tokens_per_batch, embed_dim]
        inputs = ops.reshape(inputs, [num_tokens_per_batch, self.embed_dim])
        # dispatch_tensor shape: [expert_capacity, num_experts, tokens_per_batch]
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        dispatch_tensor, combine_tensor = self.router(inputs)
        # expert_inputs shape: [num_experts, expert_capacity, embed_dim]
        expert_inputs = ops.einsum("ab,acd->cdb", inputs, dispatch_tensor)
        expert_inputs = ops.reshape(
            expert_inputs, [self.num_experts, self.expert_capacity, self.embed_dim]
        )
        # Dispatch to experts
        expert_input_list = ops.unstack(expert_inputs, axis=0)
        expert_output_list = [
            self.experts[idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
        # expert_outputs shape: [expert_capacity, num_experts, embed_dim]
        expert_outputs = ops.stack(expert_output_list, axis=1)
        # expert_outputs_combined shape: [tokens_per_batch, embed_dim]
        expert_outputs_combined = ops.einsum(
            "abc,xba->xc", expert_outputs, combine_tensor
        )
        # output shape: [batch_size, num_tokens_per_example, embed_dim]
        outputs = ops.reshape(
            expert_outputs_combined,
            [batch_size, num_tokens_per_example, self.embed_dim],
        )
        return outputs
```

## Implement a Transformer block layer {#implement-a-transformer-block-layer}

```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # The ffn can be either a standard feedforward network or a switch
        # layer with a Mixture of Experts.
        self.ffn = ffn
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
```

## Implement the classifier {#implement-the-classifier}

The `TransformerBlock` layer outputs one vector for each time step of our input sequence. Here, we take the mean across all time steps and use a feedforward network on top of it to classify text.

```python
def create_classifier():
    switch = Switch(num_experts, embed_dim, ff_dim, num_tokens_per_batch)
    transformer_block = TransformerBlock(embed_dim // num_heads, num_heads, switch)

    inputs = layers.Input(shape=(num_tokens_per_example,))
    embedding_layer = TokenAndPositionEmbedding(
        num_tokens_per_example, vocab_size, embed_dim
    )
    x = embedding_layer(inputs)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    classifier = keras.Model(inputs=inputs, outputs=outputs)
    return classifier
```

## Train and evaluate the model {#train-and-evaluate-the-model}

```python
def run_experiment(classifier):
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = classifier.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_val, y_val),
    )
    return history


classifier = create_classifier()
run_experiment(classifier)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3
 500/500 ━━━━━━━━━━━━━━━━━━━━ 251s 485ms/step - accuracy: 0.7121 - loss: 1.5394 - val_accuracy: 0.8748 - val_loss: 1.2891
Epoch 2/3
 500/500 ━━━━━━━━━━━━━━━━━━━━ 240s 480ms/step - accuracy: 0.9243 - loss: 1.2063 - val_accuracy: 0.8752 - val_loss: 1.3090
Epoch 3/3
 500/500 ━━━━━━━━━━━━━━━━━━━━ 242s 485ms/step - accuracy: 0.9572 - loss: 1.1222 - val_accuracy: 0.8614 - val_loss: 1.3744

<keras.src.callbacks.history.History at 0x7efb79d82a90>
```

{{% /details %}}

## Conclusion {#conclusion}

Compared to the standard Transformer architecture, the Switch Transformer can have a much larger number of parameters, leading to increased model capacity, while maintaining a reasonable computational cost.
