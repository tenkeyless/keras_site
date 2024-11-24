---
title: Sampler base class
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/sampler.py#L9" >}}

### `Sampler` class

`keras_nlp.samplers.Sampler(temperature=1.0)`

Base sampler class.

**Arguments**

- **temperature**: float. optional. Used to control the randomness of the sampling. The higher the temperature, the more diverse the samples. Defaults to `1.0`.

**Call arguments**

{{call\_args}}

This base class can be extended to implement different auto-regressive sampling methods. To do so, override the `get_next_token()` method, which computes the next token based on a probability distribution over all possible vocab entries.

**Example**

`causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")  # Greedy search with some tokens forbidden. class CustomSampler(keras_hub.samplers.Sampler):     def __init__(self, forbidden_tokens, **kwargs):         super().__init__(**kwargs)         self.forbidden_tokens = forbidden_tokens      def get_next_token(self, probs):         batch_size, vocab_size = keras.ops.shape(probs)         for id in self.forbidden_tokens:             update = keras.ops.zeros((batch_size, 1))             probs = keras.ops.slice_update(probs, (0, id), update)         return keras.ops.argmax(probs, axis=-1)  # 257 = "a" with a leading space, 262 = "the" with a leading space. causal_lm.compile(sampler=CustomSampler(forbidden_tokens=[257, 262])) causal_lm.summary() causal_lm.generate(["That's strange"])`

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/sampler.py#L208" >}}

### `get_next_token` method

`Sampler.get_next_token(probabilities)`

Get the next token. **Arguments**

- **probabilities**: a Tensor, the probability distribution for next token over all vocab tokens.

Get the next token based on given probability distribution over tokens. Subclasses must implement this method.

---
