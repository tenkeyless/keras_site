---
title: StochasticDepth layer
toc: true
weight: 4
type: docs
math: true
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/regularization/stochastic_depth.py#L20" >}}

### `StochasticDepth` class

```python
keras_cv.layers.StochasticDepth(rate=0.5, **kwargs)
```

Implements the Stochastic Depth layer. It randomly drops residual branches
in residual architectures. It is used as a drop-in replacement for addition
operation. Note that this layer DOES NOT drop a residual block across
individual samples but across the entire batch.

**Reference**

- [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
  - [Docstring taken from [stochastic_depth.py](https://tinyurl.com/mr3y2af6)

**Arguments**

- **rate**: float, the probability of the residual branch being dropped.

**Example**

`StochasticDepth` can be used in a residual network as follows:

```python
# (...)
input = tf.ones((1, 3, 3, 1), dtype=tf.float32)
residual = keras.layers.Conv2D(1, 1)(input)
output = keras_cv.layers.StochasticDepth()([input, residual])
# (...)
```

At train time, StochasticDepth returns:

$$
x[0] + b\_l \* x[1],
$$

where $b\_l$ is a random Bernoulli variable with probability
$P(b\_l = 1) = rate$. At test time, StochasticDepth rescales the activations
of the residual branch based on the drop rate ($rate$):

$$
x[0] + (1 - rate) \* x[1]
$$
