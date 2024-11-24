---
title: Layer weight initializers
toc: true
weight: 3
type: docs
---

## Usage of initializers

Initializers define the way to set the initial random weights of Keras layers.

The keyword arguments used for passing initializers to layers depends on the layer. Usually, it is simply `kernel_initializer` and `bias_initializer`:

`from keras import layers from keras import initializers  layer = layers.Dense(     units=64,     kernel_initializer=initializers.RandomNormal(stddev=0.01),     bias_initializer=initializers.Zeros() )`

All built-in initializers can also be passed via their string identifier:

`layer = layers.Dense(     units=64,     kernel_initializer='random_normal',     bias_initializer='zeros' )`

---

## Available initializers

The following built-in initializers are available as part of the `keras.initializers` module:

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L30)

### `RandomNormal` class

`keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)`

Random normal initializer.

Draws samples from a normal distribution for given parameters.

**Examples**

`>>> # Standalone usage: >>> initializer = RandomNormal(mean=0.0, stddev=1.0) >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = RandomNormal(mean=0.0, stddev=1.0) >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **mean**: A python scalar or a scalar keras tensor. Mean of the random values to generate.
- **stddev**: A python scalar or a scalar keras tensor. Standard deviation of the random values to generate.
- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L145)

### `RandomUniform` class

`keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)`

Random uniform initializer.

Draws samples from a uniform distribution for given parameters.

**Examples**

`>>> # Standalone usage: >>> initializer = RandomUniform(minval=0.0, maxval=1.0) >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = RandomUniform(minval=0.0, maxval=1.0) >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **minval**: A python scalar or a scalar keras tensor. Lower bound of the range of random values to generate (inclusive).
- **maxval**: A python scalar or a scalar keras tensor. Upper bound of the range of random values to generate (exclusive).
- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L86)

### `TruncatedNormal` class

`keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)`

Initializer that generates a truncated normal distribution.

The values generated are similar to values from a `RandomNormal` initializer, except that values more than two standard deviations from the mean are discarded and re-drawn.

**Examples**

`>>> # Standalone usage: >>> initializer = TruncatedNormal(mean=0., stddev=1.) >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = TruncatedNormal(mean=0., stddev=1.) >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **mean**: A python scalar or a scalar keras tensor. Mean of the random values to generate.
- **stddev**: A python scalar or a scalar keras tensor. Standard deviation of the random values to generate.
- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/constant_initializers.py#L48)

### `Zeros` class

`keras.initializers.Zeros()`

Initializer that generates tensors initialized to 0.

**Examples**

`>>> # Standalone usage: >>> initializer = Zeros() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = Zeros() >>> layer = Dense(units=3, kernel_initializer=initializer)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/constant_initializers.py#L77)

### `Ones` class

`keras.initializers.Ones()`

Initializer that generates tensors initialized to 1.

Also available via the shortcut function `ones`.

**Examples**

`>>> # Standalone usage: >>> initializer = Ones() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = Ones() >>> layer = Dense(3, kernel_initializer=initializer)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L369)

### `GlorotNormal` class

`keras.initializers.GlorotNormal(seed=None)`

The Glorot normal initializer, also called Xavier normal initializer.

Draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units in the weight tensor.

**Examples**

`>>> # Standalone usage: >>> initializer = GlorotNormal() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = GlorotNormal() >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

**Reference**

- [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L320)

### `GlorotUniform` class

`keras.initializers.GlorotUniform(seed=None)`

The Glorot uniform initializer, also called Xavier uniform initializer.

Draws samples from a uniform distribution within `[-limit, limit]`, where `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units).

**Examples**

`>>> # Standalone usage: >>> initializer = GlorotUniform() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = GlorotUniform() >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

**Reference**

- [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L524)

### `HeNormal` class

`keras.initializers.HeNormal(seed=None)`

He normal initializer.

It draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in the weight tensor.

**Examples**

`>>> # Standalone usage: >>> initializer = HeNormal() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = HeNormal() >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

**Reference**

- [He et al., 2015](https://arxiv.org/abs/1502.01852)

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L568)

### `HeUniform` class

`keras.initializers.HeUniform(seed=None)`

He uniform variance scaling initializer.

Draws samples from a uniform distribution within `[-limit, limit]`, where `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the weight tensor).

**Examples**

`>>> # Standalone usage: >>> initializer = HeUniform() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = HeUniform() >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

**Reference**

- [He et al., 2015](https://arxiv.org/abs/1502.01852)

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L640)

### `OrthogonalInitializer` class

`keras.initializers.Orthogonal(gain=1.0, seed=None)`

Initializer that generates an orthogonal matrix.

If the shape of the tensor to initialize is two-dimensional, it is initialized with an orthogonal matrix obtained from the QR decomposition of a matrix of random numbers drawn from a normal distribution. If the matrix has fewer rows than columns then the output will have orthogonal rows. Otherwise, the output will have orthogonal columns.

If the shape of the tensor to initialize is more than two-dimensional, a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])` is initialized, where `n` is the length of the shape vector. The matrix is subsequently reshaped to give a tensor of the desired shape.

**Examples**

`>>> # Standalone usage: >>> initializer = keras.initializers.Orthogonal() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = keras.initializers.Orthogonal() >>> layer = keras.layers.Dense(3, kernel_initializer=initializer)`

**Arguments**

- **gain**: Multiplicative factor to apply to the orthogonal matrix.
- **seed**: A Python integer. Used to make the behavior of the initializer deterministic.

**Reference**

- [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/constant_initializers.py#L8)

### `Constant` class

`keras.initializers.Constant(value=0.0)`

Initializer that generates tensors with constant values.

Only scalar values are allowed. The constant value provided must be convertible to the dtype requested when calling the initializer.

**Examples**

`>>> # Standalone usage: >>> initializer = Constant(10.) >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = Constant(10.) >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **value**: A Python scalar.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L201)

### `VarianceScaling` class

`keras.initializers.VarianceScaling(     scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None )`

Initializer that adapts its scale to the shape of its input tensors.

With `distribution="truncated_normal" or "untruncated_normal"`, samples are drawn from a truncated/untruncated normal distribution with a mean of zero and a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`, where `n` is:

- number of input units in the weight tensor, if `mode="fan_in"`
- number of output units, if `mode="fan_out"`
- average of the numbers of input and output units, if `mode="fan_avg"`

With `distribution="uniform"`, samples are drawn from a uniform distribution within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

**Examples**

`>>> # Standalone usage: >>> initializer = VarianceScaling(     scale=0.1, mode='fan_in', distribution='uniform') >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = VarianceScaling(     scale=0.1, mode='fan_in', distribution='uniform') >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **scale**: Scaling factor (positive float).
- **mode**: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
- **distribution**: Random distribution to use. One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L422)

### `LecunNormal` class

`keras.initializers.LecunNormal(seed=None)`

Lecun normal initializer.

Initializers allow you to pre-specify an initialization strategy, encoded in the Initializer object, without knowing the shape and dtype of the variable being initialized.

Draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in the weight tensor.

**Examples**

`>>> # Standalone usage: >>> initializer = LecunNormal() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = LecunNormal() >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

**Reference**

- [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/random_initializers.py#L475)

### `LecunUniform` class

`keras.initializers.LecunUniform(seed=None)`

Lecun uniform initializer.

Draws samples from a uniform distribution within `[-limit, limit]`, where `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the weight tensor).

**Examples**

`>>> # Standalone usage: >>> initializer = LecunUniform() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = LecunUniform() >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **seed**: A Python integer or instance of `keras.backend.SeedGenerator`. Used to make the behavior of the initializer deterministic. Note that an initializer seeded with an integer or `None` (unseeded) will produce the same random values across multiple calls. To get different random values across multiple calls, use as seed an instance of `keras.backend.SeedGenerator`.

**Reference**

- [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/initializers/constant_initializers.py#L108)

### `Identity` class

`keras.initializers.IdentityInitializer(gain=1.0)`

Initializer that generates the identity matrix.

Only usable for generating 2D matrices.

**Examples**

`>>> # Standalone usage: >>> initializer = Identity() >>> values = initializer(shape=(2, 2))`

`>>> # Usage in a Keras layer: >>> initializer = Identity() >>> layer = Dense(3, kernel_initializer=initializer)`

**Arguments**

- **gain**: Multiplicative factor to apply to the identity matrix.

---

## Creating custom initializers

### Simple callables

You can pass a custom callable as initializer. It must take the arguments `shape` (shape of the variable to initialize) and `dtype` (dtype of generated values):

`def my_init(shape, dtype=None):     return keras.random.normal(shape, dtype=dtype)  layer = Dense(64, kernel_initializer=my_init)`

### `Initializer` subclasses

If you need to configure your initializer via various arguments (e.g. `stddev` argument in `RandomNormal`), you should implement it as a subclass of `keras.initializers.Initializer`.

Initializers should implement a `__call__` method with the following signature:

`` def __call__(self, shape, dtype=None)`:     # returns a tensor of shape `shape` and dtype `dtype`     # containing values drawn from a distribution of your choice. ``

Optionally, you an also implement the method `get_config` and the class method `from_config` in order to support serialization – just like with any Keras object.

Here's a simple example: a random normal initializer.

`` class ExampleRandomNormal(keras.initializers.Initializer):      def __init__(self, mean, stddev):       self.mean = mean       self.stddev = stddev      def __call__(self, shape, dtype=None)`:       return keras.random.normal(           shape, mean=self.mean, stddev=self.stddev, dtype=dtype)      def get_config(self):  # To support serialization       return {'mean': self.mean, 'stddev': self.stddev} ``

Note that we don't have to implement `from_config` in the example above since the constructor arguments of the class the keys in the config returned by `get_config` are the same. In this case, the default `from_config` works fine.
