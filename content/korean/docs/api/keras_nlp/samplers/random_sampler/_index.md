---
title: RandomSampler
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/random_sampler.py#L8" >}}

### `RandomSampler` class

```python
keras_nlp.samplers.RandomSampler(seed=None, **kwargs)
```

Random Sampler class.

This sampler implements random sampling. Briefly, random sampler randomly
selects a token from the entire distribution of the tokens, with selection
chance determined by the probability of each token.

**Arguments**

- **seed**: int. The random seed. Defaults to `None`.

**Call arguments**

{{call\_args}}

**Examples**

```python
causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
# Pass by name to compile.
causal_lm.compile(sampler="random")
causal_lm.generate(["Keras is a"])
# Pass by object to compile.
sampler = keras_hub.samplers.RandomSampler(temperature=0.7)
causal_lm.compile(sampler=sampler)
causal_lm.generate(["Keras is a"])
```
