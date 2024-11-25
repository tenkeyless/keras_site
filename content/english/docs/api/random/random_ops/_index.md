---
title: Random operations
toc: true
weight: 2
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L272" >}}

### `beta` function

`keras.random.beta(shape, alpha, beta, dtype=None, seed=None)`

Draw samples from a Beta distribution.

The values are drawm from a Beta distribution parametrized by alpha and beta.

**Arguments**

- **shape**: The shape of the random values to generate.
- **alpha**: Float or an array of floats representing the first parameter alpha. Must be broadcastable with `beta` and `shape`.
- **beta**: Float or an array of floats representing the second parameter beta. Must be broadcastable with `alpha` and `shape`.
- **dtype**: Optional dtype of the tensor. Only floating point types are supported. If not specified, `keras.config.floatx()` is used, which defaults to `float32` unless you configured it otherwise (via `keras.config.set_floatx(float_dtype)`).
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L236" >}}

### `binomial` function

`keras.random.binomial(shape, counts, probabilities, dtype=None, seed=None)`

Draw samples from a Binomial distribution.

The values are drawn from a Binomial distribution with specified trial count and probability of success.

**Arguments**

- **shape**: The shape of the random values to generate.
- **counts**: A number or array of numbers representing the number of trials. It must be broadcastable with `probabilities`.
- **probabilities**: A float or array of floats representing the probability of success of an individual event. It must be broadcastable with `counts`.
- **dtype**: Optional dtype of the tensor. Only floating point types are supported. If not specified, `keras.config.floatx()` is used, which defaults to `float32` unless you configured it otherwise (via `keras.config.set_floatx(float_dtype)`).
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L32" >}}

### `categorical` function

`keras.random.categorical(logits, num_samples, dtype="int32", seed=None)`

Draws samples from a categorical distribution.

This function takes as input `logits`, a 2-D input tensor with shape (batch_size, num_classes). Each row of the input represents a categorical distribution, with each column index containing the log-probability for a given class.

The function will output a 2-D tensor with shape (batch_size, num_samples), where each row contains samples from the corresponding row in `logits`. Each column index contains an independent samples drawn from the input distribution.

**Arguments**

- **logits**: 2-D Tensor with shape (batch_size, num_classes). Each row should define a categorical distribution with the unnormalized log-probabilities for all classes.
- **num_samples**: Int, the number of independent samples to draw for each row of the input. This will be the second dimension of the output tensor's shape.
- **dtype**: Optional dtype of the output tensor.
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

**Returns**

A 2-D tensor with (batch_size, num_samples).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L186" >}}

### `dropout` function

`keras.random.dropout(inputs, rate, noise_shape=None, seed=None)`

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L213" >}}

### `gamma` function

`keras.random.gamma(shape, alpha, dtype=None, seed=None)`

Draw random samples from the Gamma distribution.

**Arguments**

- **shape**: The shape of the random values to generate.
- **alpha**: Float, the parameter of the distribution.
- **dtype**: Optional dtype of the tensor. Only floating point types are supported. If not specified, `keras.config.floatx()` is used, which defaults to `float32` unless you configured it otherwise (via `keras.config.set_floatx(float_dtype)`).
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L5" >}}

### `normal` function

`keras.random.normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)`

Draw random samples from a normal (Gaussian) distribution.

**Arguments**

- **shape**: The shape of the random values to generate.
- **mean**: Float, defaults to 0. Mean of the random values to generate.
- **stddev**: Float, defaults to 1. Standard deviation of the random values to generate.
- **dtype**: Optional dtype of the tensor. Only floating point types are supported. If not specified, `keras.config.floatx()` is used, which defaults to `float32` unless you configured it otherwise (via `keras.config.set_floatx(float_dtype)`).
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L116" >}}

### `randint` function

`keras.random.randint(shape, minval, maxval, dtype="int32", seed=None)`

Draw random integers from a uniform distribution.

The generated values follow a uniform distribution in the range `[minval, maxval)`. The lower bound `minval` is included in the range, while the upper bound `maxval` is excluded.

`dtype` must be an integer type.

**Arguments**

- **shape**: The shape of the random values to generate.
- **minval**: Float, defaults to 0. Lower bound of the range of random values to generate (inclusive).
- **maxval**: Float, defaults to 1. Upper bound of the range of random values to generate (exclusive).
- **dtype**: Optional dtype of the tensor. Only integer types are supported. If not specified, `keras.config.floatx()` is used, which defaults to `float32` unless you configured it otherwise (via `keras.config.set_floatx(float_dtype)`)
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L193" >}}

### `shuffle` function

`keras.random.shuffle(x, axis=0, seed=None)`

Shuffle the elements of a tensor uniformly at random along an axis.

**Arguments**

- **x**: The tensor to be shuffled.
- **axis**: An integer specifying the axis along which to shuffle. Defaults to `0`.
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L155" >}}

### `truncated_normal` function

`keras.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)`

Draw samples from a truncated normal distribution.

The values are drawn from a normal distribution with specified mean and standard deviation, discarding and re-drawing any samples that are more than two standard deviations from the mean.

**Arguments**

- **shape**: The shape of the random values to generate.
- **mean**: Float, defaults to 0. Mean of the random values to generate.
- **stddev**: Float, defaults to 1. Standard deviation of the random values to generate.
- **dtype**: Optional dtype of the tensor. Only floating point types are supported. If not specified, `keras.config.floatx()` is used, which defaults to `float32` unless you configured it otherwise (via `keras.config.set_floatx(float_dtype)`)
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/random.py#L77" >}}

### `uniform` function

`keras.random.uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)`

Draw samples from a uniform distribution.

The generated values follow a uniform distribution in the range `[minval, maxval)`. The lower bound `minval` is included in the range, while the upper bound `maxval` is excluded.

`dtype` must be a floating point type, the default range is `[0, 1)`.

**Arguments**

- **shape**: The shape of the random values to generate.
- **minval**: Float, defaults to 0. Lower bound of the range of random values to generate (inclusive).
- **maxval**: Float, defaults to 1. Upper bound of the range of random values to generate (exclusive).
- **dtype**: Optional dtype of the tensor. Only floating point types are supported. If not specified, `keras.config.floatx()` is used, which defaults to `float32` unless you configured it otherwise (via `keras.config.set_floatx(float_dtype)`)
- **seed**: A Python integer or instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class). Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or None (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of [`keras.random.SeedGenerator`](/api/random/seed_generator#seedgenerator-class).

---
