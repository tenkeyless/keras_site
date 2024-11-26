---
title: TopKSampler
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/top_k_sampler.py#L8" >}}

### `TopKSampler` class

```python
keras_nlp.samplers.TopKSampler(k=5, seed=None, **kwargs)
```

Top-K Sampler class.

This sampler implements top-k search algorithm. Briefly, top-k algorithm
randomly selects a token from the tokens of top K probability, with
selection chance determined by the probability.

**Arguments**

- **k**: int, the `k` value of top-k.
- **seed**: int. The random seed. Defaults to `None`.

**Call arguments**

{{call\_args}}

**Examples**

```python
causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
# Pass by name to compile.
causal_lm.compile(sampler="top_k")
causal_lm.generate(["Keras is a"])
# Pass by object to compile.
sampler = keras_hub.samplers.TopKSampler(k=5, temperature=0.7)
causal_lm.compile(sampler=sampler)
causal_lm.generate(["Keras is a"])
```
