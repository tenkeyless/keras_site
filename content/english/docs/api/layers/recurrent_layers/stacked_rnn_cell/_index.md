---
title: Stacked RNN cell layer
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/rnn/stacked_rnn_cells.py#L8)

### `StackedRNNCells` class

```python
keras.layers.StackedRNNCells(cells, **kwargs)
```

Wrapper allowing a stack of RNN cells to behave as a single cell.

Used to implement efficient stacked RNNs.

**Arguments**

- **cells**: List of RNN cell instances.

**Example**

```python
batch_size = 3
sentence_length = 5
num_features = 2
new_shape = (batch_size, sentence_length, num_features)
x = np.reshape(np.arange(30), new_shape)

rnn_cells = [keras.layers.LSTMCell(128) for _ in range(2)]
stacked_lstm = keras.layers.StackedRNNCells(rnn_cells)
lstm_layer = keras.layers.RNN(stacked_lstm)

result = lstm_layer(x)
```
