---
title: ContrastiveSampler
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/contrastive_sampler.py#L9" >}}

### `ContrastiveSampler` class

```python
keras_nlp.samplers.ContrastiveSampler(k=5, alpha=0.6, **kwargs)
```

Contrastive Sampler class.

This sampler implements contrastive search algorithm. In short, the sampler
chooses the token having the max "score" as the next token. The "score" is
a weighted sum between token's probability and max similarity against
previous tokens. By using this joint score, contrastive sampler reduces the
behavior of duplicating seen tokens.

**Arguments**

- **k**: int, the `k` value of top-k. Next token will be chosen from k tokens.
- **alpha**: float, the weight of minus max similarity in joint score
  computation. The larger the value of `alpha`, the score relies more
  on the similarity than the token probability.

**Call arguments**

{{call\_args}}

**Examples**

```python
causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
# Pass by name to compile.
causal_lm.compile(sampler="contrastive")
causal_lm.generate(["Keras is a"])
# Pass by object to compile.
sampler = keras_hub.samplers.ContrastiveSampler(k=5)
causal_lm.compile(sampler=sampler)
causal_lm.generate(["Keras is a"])
```
