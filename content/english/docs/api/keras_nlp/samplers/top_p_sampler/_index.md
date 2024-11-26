---
title: TopPSampler
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/top_p_sampler.py#L8" >}}

### `TopPSampler` class

```python
keras_nlp.samplers.TopPSampler(p=0.1, k=None, seed=None, **kwargs)
```

Top-P Sampler class.

This sampler implements top-p search algorithm. Top-p search selects tokens
from the smallest subset of output probabilities that sum to greater than
`p`. Put in another way, top-p will first order token predictions by
likelihood, and ignore all tokens after the cumulative probability of
selected tokens exceeds `p`, then select a token from the remaining tokens.

**Arguments**

- **p**: float, the `p` value of top-p.
- **k**: int. If set, this argument defines a
  heuristic "top-k" cutoff applied before the "top-p" sampling. All
  logits not in the top `k` will be discarded, and the remaining
  logits will be sorted to find a cutoff point for `p`. Setting this
  arg can significantly speed sampling up by reducing the number
  of tokens to sort. Defaults to `None`.
- **seed**: int. The random seed. Defaults to `None`.

**Call arguments**

{{call\_args}}

**Examples**

```python
causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
# Pass by name to compile.
causal_lm.compile(sampler="top_p")
causal_lm.generate(["Keras is a"])
# Pass by object to compile.
sampler = keras_hub.samplers.TopPSampler(p=0.1, k=1_000)
causal_lm.compile(sampler=sampler)
causal_lm.generate(["Keras is a"])
```
