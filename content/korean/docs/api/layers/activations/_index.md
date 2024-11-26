---
title: Layer activation functions
linkTitle: Layer activations
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

## Usage of activations

Activations can either be used through an `Activation` layer, or through the `activation` argument supported by all forward layers:

```python
model.add(layers.Dense(64, activation=activations.relu))
```

This is equivalent to:

```python
from keras import layers
from keras import activations

model.add(layers.Dense(64))
model.add(layers.Activation(activations.relu))
```

All built-in activations may also be passed via their string identifier:

```python
model.add(layers.Dense(64, activation='relu'))
```

## Available activations

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L6" >}}

### `relu` function

```python
keras.activations.relu(x, negative_slope=0.0, max_value=None, threshold=0.0)
```

Applies the rectified linear unit activation function.

With default values, this returns the standard ReLU activation: `max(x, 0)`, the element-wise maximum of 0 and the input tensor.

Modifying default parameters allows you to use non-zero thresholds, change the max value of the activation, and to use a non-zero multiple of the input for values below the threshold.

**Examples**

```console
>>> x = [-10, -5, 0.0, 5, 10]
>>> keras.activations.relu(x)
[ 0.,  0.,  0.,  5., 10.]
>>> keras.activations.relu(x, negative_slope=0.5)
[-5. , -2.5,  0. ,  5. , 10. ]
>>> keras.activations.relu(x, max_value=5.)
[0., 0., 0., 5., 5.]
>>> keras.activations.relu(x, threshold=5.)
[-0., -0.,  0.,  0., 10.]
```

**Arguments**

- **x**: Input tensor.
- **negative_slope**: A `float` that controls the slope for values lower than the threshold.
- **max_value**: A `float` that sets the saturation threshold (the largest value the function will return).
- **threshold**: A `float` giving the threshold value of the activation function below which values will be damped or set to zero.

**Returns**

A tensor with the same shape and dtype as input `x`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L317" >}}

### `sigmoid` function

```python
keras.activations.sigmoid(x)
```

Sigmoid activation function.

It is defined as: `sigmoid(x) = 1 / (1 + exp(-x))`.

For small values (<-5), `sigmoid` returns a value close to zero, and for large values (>5) the result of the function gets close to 1.

Sigmoid is equivalent to a 2-element softmax, where the second element is assumed to be zero. The sigmoid function always returns a value between 0 and 1.

**Arguments**

- **x**: Input tensor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L135" >}}

### `softmax` function

```python
keras.activations.softmax(x, axis=-1)
```

Softmax converts a vector of values to a probability distribution.

The elements of the output vector are in range `[0, 1]` and sum to 1.

Each input vector is handled independently. The `axis` argument sets which axis of the input the function is applied along.

Softmax is often used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.

The softmax of each vector x is computed as `exp(x) / sum(exp(x))`.

The input values in are the log-odds of the resulting probability.

**Arguments**

- **x**: Input tensor.
- **axis**: Integer, axis along which the softmax is applied.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L236" >}}

### `softplus` function

```python
keras.activations.softplus(x)
```

Softplus activation function.

It is defined as: `softplus(x) = log(exp(x) + 1)`.

**Arguments**

- **x**: Input tensor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L248" >}}

### `softsign` function

```python
keras.activations.softsign(x)
```

Softsign activation function.

Softsign is defined as: `softsign(x) = x / (abs(x) + 1)`.

**Arguments**

- **x**: Input tensor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L303" >}}

### `tanh` function

```python
keras.activations.tanh(x)
```

Hyperbolic tangent activation function.

It is defined as: `tanh(x) = sinh(x) / cosh(x)`, i.e. `tanh(x) = ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))`.

**Arguments**

- **x**: Input tensor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L196" >}}

### `selu` function

```python
keras.activations.selu(x)
```

Scaled Exponential Linear Unit (SELU).

The Scaled Exponential Linear Unit (SELU) activation function is defined as:

- `scale * x` if `x > 0`
- `scale * alpha * (exp(x) - 1)` if `x < 0`

where `alpha` and `scale` are pre-defined constants (`alpha=1.67326324` and `scale=1.05070098`).

Basically, the SELU activation function multiplies `scale` (> 1) with the output of the [`keras.activations.elu`]({{< relref "/docs/api/layers/activations#elu-function" >}}) function to ensure a slope larger than one for positive inputs.

The values of `alpha` and `scale` are chosen so that the mean and variance of the inputs are preserved between two consecutive layers as long as the weights are initialized correctly (see [`keras.initializers.LecunNormal`]({{< relref "/docs/api/layers/initializers#lecunnormal-class" >}}) initializer) and the number of input units is "large enough" (see reference paper for more information).

**Arguments**

- **x**: Input tensor.

Notes:

- To be used together with the [`keras.initializers.LecunNormal`]({{< relref "/docs/api/layers/initializers#lecunnormal-class" >}}) initializer.
- To be used together with the dropout variant [`keras.layers.AlphaDropout`]({{< relref "/docs/api/layers/regularization_layers/alpha_dropout#alphadropout-class" >}}) (rather than regular dropout).

**Reference**

- [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L168" >}}

### `elu` function

```python
keras.activations.elu(x, alpha=1.0)
```

Exponential Linear Unit.

The exponential linear unit (ELU) with `alpha > 0` is defined as:

- `x` if `x > 0`
- alpha \* `exp(x) - 1` if `x < 0`

ELUs have negative values which pushes the mean of the activations closer to zero.

Mean activations that are closer to zero enable faster learning as they bring the gradient closer to the natural gradient. ELUs saturate to a negative value when the argument gets smaller. Saturation means a small derivative which decreases the variation and the information that is propagated to the next layer.

**Arguments**

- **x**: Input tensor.

**Reference**

- [Clevert et al., 2016](https://arxiv.org/abs/1511.07289)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L344" >}}

### `exponential` function

```python
keras.activations.exponential(x)
```

Exponential activation function.

**Arguments**

- **x**: Input tensor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L111" >}}

### `leaky_relu` function

```python
keras.activations.leaky_relu(x, negative_slope=0.2)
```

Leaky relu activation function.

**Arguments**

- **x**: Input tensor.
- **negative_slope**: A `float` that controls the slope for values lower than the threshold.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L123" >}}

### `relu6` function

```python
keras.activations.relu6(x)
```

Relu6 activation function.

It's the ReLU function, but truncated to a maximum value of 6.

**Arguments**

- **x**: Input tensor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L260" >}}

### `silu` function

```python
keras.activations.silu(x)
```

Swish (or Silu) activation function.

It is defined as: `swish(x) = x * sigmoid(x)`.

The Swish (or Silu) activation function is a smooth, non-monotonic function that is unbounded above and bounded below.

**Arguments**

- **x**: Input tensor.

**Reference**

- [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L377" >}}

### `hard_silu` function

```python
keras.activations.hard_silu(x)
```

Hard SiLU activation function, also known as Hard Swish.

It is defined as:

- `0` if `if x < -3`
- `x` if `x > 3`
- `x * (x + 3) / 6` if `-3 <= x <= 3`

It's a faster, piecewise linear approximation of the silu activation.

**Arguments**

- **x**: Input tensor.

**Reference**

- [A Howard, 2019](https://arxiv.org/abs/1905.02244)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L280" >}}

### `gelu` function

```python
keras.activations.gelu(x, approximate=False)
```

Gaussian error linear unit (GELU) activation function.

The Gaussian error linear unit (GELU) is defined as:

`gelu(x) = x * P(X <= x)` where `P(X) ~ N(0, 1)`, i.e. `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.

GELU weights inputs by their value, rather than gating inputs by their sign as in ReLU.

**Arguments**

- **x**: Input tensor.
- **approximate**: A `bool`, whether to enable approximation.

**Reference**

- [Hendrycks et al., 2016](https://arxiv.org/abs/1606.08415)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L354" >}}

### `hard_sigmoid` function

```python
keras.activations.hard_sigmoid(x)
```

Hard sigmoid activation function.

The hard sigmoid activation is defined as:

- `0` if `if x <= -3`
- `1` if `x >= 3`
- `(x/6) + 0.5` if `-3 < x < 3`

It's a faster, piecewise linear approximation of the sigmoid activation.

**Arguments**

- **x**: Input tensor.

**Reference**

- [Wikipedia "Hard sigmoid"](https://en.wikipedia.org/wiki/Hard_sigmoid)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L400" >}}

### `linear` function

```python
keras.activations.linear(x)
```

Linear activation function (pass-through).

A "linear" activation is an identity function: it returns the input, unmodified.

**Arguments**

- **x**: Input tensor.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L425" >}}

### `mish` function

```python
keras.activations.mish(x)
```

Mish activation function.

It is defined as:

`mish(x) = x * tanh(softplus(x))`

where `softplus` is defined as:

`softplus(x) = log(exp(x) + 1)`

**Arguments**

- **x**: Input tensor.

**Reference**

- [Misra, 2019](https://arxiv.org/abs/1908.08681)

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/activations/activations.py#L448" >}}

### `log_softmax` function

```python
keras.activations.log_softmax(x, axis=-1)
```

Log-Softmax activation function.

Each input vector is handled independently. The `axis` argument sets which axis of the input the function is applied along.

**Arguments**

- **x**: Input tensor.
- **axis**: Integer, axis along which the softmax is applied.

## Creating custom activations

You can also use a callable as an activation (in this case it should take a tensor and return a tensor of the same shape and dtype):

```python
model.add(layers.Dense(64, activation=keras.ops.tanh))
```

## About "advanced activation" layers

Activations that are more complex than a simple function (eg. learnable activations, which maintain a state) are available as [Advanced Activation layers]({{< relref "/docs/api/layers/activation_layers/" >}}).
