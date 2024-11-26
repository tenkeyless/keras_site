---
title: GreedySampler
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/greedy_sampler.py#L7" >}}

### `GreedySampler` class

```python
keras_nlp.samplers.GreedySampler(**kwargs)
```

Greedy sampler class.

This sampler is implemented on greedy search, i.e., always picking up the
token of the largest probability as the next token.

**Examples**

```python
causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
# Pass by name to compile.
causal_lm.compile(sampler="greedy")
causal_lm.generate(["Keras is a"])
# Pass by object to compile.
sampler = keras_hub.samplers.GreedySampler()
causal_lm.compile(sampler=sampler)
causal_lm.generate(["Keras is a"])
```
