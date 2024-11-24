---
title: Bidirectional LSTM on IMDB
toc: true
weight: 9
type: docs
---

{{< original checkedAt="2024-11-21" >}}

**Author:** [fchollet](https://twitter.com/fchollet)  
**Date created:** 2020/05/03  
**Last modified:** 2020/05/03  
**Description:** Train a 2-layer bidirectional LSTM on the IMDB movie review sentiment classification dataset.

{{< hextra/hero-button
    text="ⓘ This example uses Keras 3"
    style="background: rgb(23, 132, 133); margin: 1em 0 0.5em 0; pointer-events: none;" >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/bidirectional_lstm_imdb.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/nlp/bidirectional_lstm_imdb.py" title="GitHub source" tag="GitHub">}}
{{< /cards >}}

## Setup

```python
import numpy as np
import keras
from keras import layers

max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
```

## Build the model

```python
# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()
```

{{% details title="Result" closed="true" %}}

```plain
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, None)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ embedding (Embedding)           │ (None, None, 128)         │  2,560,000 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ bidirectional (Bidirectional)   │ (None, None, 128)         │     98,816 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ bidirectional_1 (Bidirectional) │ (None, 128)               │     98,816 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 1)                 │        129 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 2,757,761 (10.52 MB)
 Trainable params: 2,757,761 (10.52 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

## Load the IMDB movie review sentiment data

```python
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
# Use pad_sequence to standardize sequence length:
# this will truncate sequences longer than 200 words and zero-pad sequences shorter than 200 words.
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)
```

{{% details title="Result" closed="true" %}}

```plain
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
 17464789/17464789 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
25000 Training sequences
25000 Validation sequences
```

{{% /details %}}

## Train and evaluate the model

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/bidirectional-lstm-imdb) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/bidirectional_lstm_imdb).

```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
```

{{% details title="Result" closed="true" %}}

```plain
Epoch 1/2
 782/782 ━━━━━━━━━━━━━━━━━━━━ 61s 75ms/step - accuracy: 0.7540 - loss: 0.4697 - val_accuracy: 0.8269 - val_loss: 0.4202
Epoch 2/2
 782/782 ━━━━━━━━━━━━━━━━━━━━ 54s 69ms/step - accuracy: 0.9151 - loss: 0.2263 - val_accuracy: 0.8428 - val_loss: 0.3650

<keras.src.callbacks.history.History at 0x7f3efd663850>
```

{{% /details %}}
