---
title: Simple RNN cell layer
toc: true
weight: 12
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/rnn/simple_rnn.py#L14" >}}

### `SimpleRNNCell` class

```python
keras.layers.SimpleRNNCell(
    units,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    seed=None,
    **kwargs
)
```

Cell class for SimpleRNN.

This class processes one step within the whole time sequence input, whereas `keras.layer.SimpleRNN` processes the whole sequence.

**Arguments**

- **units**: Positive integer, dimensionality of the output space.
- **activation**: Activation function to use. Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
- **use_bias**: Boolean, (default `True`), whether the layer should use a bias vector.
- **kernel_initializer**: Initializer for the `kernel` weights matrix, used for the linear transformation of the inputs. Default: `"glorot_uniform"`.
- **recurrent_initializer**: Initializer for the `recurrent_kernel` weights matrix, used for the linear transformation of the recurrent state. Default: `"orthogonal"`.
- **bias_initializer**: Initializer for the bias vector. Default: `"zeros"`.
- **kernel_regularizer**: Regularizer function applied to the `kernel` weights matrix. Default: `None`.
- **recurrent_regularizer**: Regularizer function applied to the `recurrent_kernel` weights matrix. Default: `None`.
- **bias_regularizer**: Regularizer function applied to the bias vector. Default: `None`.
- **kernel_constraint**: Constraint function applied to the `kernel` weights matrix. Default: `None`.
- **recurrent_constraint**: Constraint function applied to the `recurrent_kernel` weights matrix. Default: `None`.
- **bias_constraint**: Constraint function applied to the bias vector. Default: `None`.
- **dropout**: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default: 0.
- **recurrent_dropout**: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.
- **seed**: Random seed for dropout.

**Call arguments**

- **sequence**: A 2D tensor, with shape `(batch, features)`.
- **states**: A 2D tensor with shape `(batch, units)`, which is the state from the previous time step.
- **training**: Python boolean indicating whether the layer should behave in training mode or in inference mode. Only relevant when `dropout` or `recurrent_dropout` is used.

**Example**

```python
inputs = np.random.random([32, 10, 8]).astype(np.float32)
rnn = keras.layers.RNN(keras.layers.SimpleRNNCell(4))
output = rnn(inputs)  # The output has shape `(32, 4)`.
rnn = keras.layers.RNN(
    keras.layers.SimpleRNNCell(4),
    return_sequences=True,
    return_state=True
)
# whole_sequence_output has shape `(32, 10, 4)`.
# final_state has shape `(32, 4)`.
whole_sequence_output, final_state = rnn(inputs)
```
