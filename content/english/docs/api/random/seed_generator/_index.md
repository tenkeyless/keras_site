---
title: SeedGenerator class
toc: true
weight: 1
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/random/seed_generator.py#L12" >}}

### `SeedGenerator` class

`keras.random.SeedGenerator(seed=None, name=None, **kwargs)`

Generates variable seeds upon each call to a RNG-using function.

In Keras, all RNG-using methods (such as `keras.random.normal()`) are stateless, meaning that if you pass an integer seed to them (such as `seed=42`), they will return the same values at each call. In order to get different values at each call, you must use a `SeedGenerator` instead as the seed argument. The `SeedGenerator` object is stateful.

**Example**

`seed_gen = keras.random.SeedGenerator(seed=42) values = keras.random.normal(shape=(2, 3), seed=seed_gen) new_values = keras.random.normal(shape=(2, 3), seed=seed_gen)`

Usage in a layer:

`class Dropout(keras.Layer):     def __init__(self, **kwargs):         super().__init__(**kwargs)         self.seed_generator = keras.random.SeedGenerator(1337)      def call(self, x, training=False):         if training:             return keras.random.dropout(                 x, rate=0.5, seed=self.seed_generator             )         return x`

---
