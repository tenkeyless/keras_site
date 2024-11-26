---
title: BeamSampler
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/samplers/beam_sampler.py#L10" >}}

### `BeamSampler` class

```python
keras_hub.samplers.BeamSampler(num_beams=5, return_all_beams=False, **kwargs)
```

Beam Sampler class.

This sampler implements beam search algorithm. At each time-step, beam
search keeps the beams (sequences) of the top `num_beams` highest
accumulated probabilities, and uses each one of the beams to predict
candidate next tokens.

**Arguments**

- **num_beams**: int. The number of beams that should be kept at each
  time-step. `num_beams` should be strictly positive.
- **return_all_beams**: bool. When set to `True`, the sampler will return all
  beams and their respective probabilities score.

**Call arguments**

{{call\_args}}

**Examples**

```python
causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
# Pass by name to compile.
causal_lm.compile(sampler="beam")
causal_lm.generate(["Keras is a"])
# Pass by object to compile.
sampler = keras_hub.samplers.BeamSampler(num_beams=5)
causal_lm.compile(sampler=sampler)
causal_lm.generate(["Keras is a"])
```
