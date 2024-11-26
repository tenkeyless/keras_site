---
title: FalconCausalLM model
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/falcon/falcon_causal_lm.py#L12" >}}

### `FalconCausalLM` class

```python
keras_nlp.models.FalconCausalLM(backbone, preprocessor=None, **kwargs)
```

An end-to-end Falcon model for causal language modeling.

A causal language model (LM) predicts the next token based on previous
tokens. This task setup can be used to train the model unsupervised on
plain text input, or to autoregressively generate plain text similar to
the data used for training. This task can be used for pre-training or
fine-tuning a Falcon model, simply by calling `fit()`.

This model has a `generate()` method, which generates text based on a
prompt. The generation strategy used is controlled by an additional
`sampler` argument on `compile()`. You can recompile the model with
different `keras_hub.samplers` objects to control the generation. By
default, `"greedy"` sampling will be used.

This model can optionally be configured with a `preprocessor` layer, in
which case it will automatically apply preprocessing to string inputs during
`fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
when creating the model with `from_preset()`.

**Arguments**

- **backbone**: A [`keras_hub.models.FalconBackbone`]({{< relref "/docs/api/keras_hub/models/falcon/falcon_backbone#falconbackbone-class" >}}) instance.
- **preprocessor**: A [`keras_hub.models.FalconCausalLMPreprocessor`]({{< relref "/docs/api/keras_hub/models/falcon/falcon_causal_lm_preprocessor#falconcausallmpreprocessor-class" >}}) or `None`.
  If `None`, this model will not apply preprocessing, and inputs
  should be preprocessed before calling the model.

**Examples**

Use `generate()` to do text generation.

```python
falcon_lm = keras_hub.models.FalconCausalLM.from_preset("falcon_refinedweb_1b_en")
falcon_lm.generate("I want to say", max_length=30)
# Generate with batched prompts.
falcon_lm.generate(["This is a", "Where are you"], max_length=30)
```

Compile the `generate()` function with a custom sampler.

```python
falcon_lm = keras_hub.models.FalconCausalLM.from_preset("falcon_refinedweb_1b_en")
falcon_lm.compile(sampler="top_k")
falcon_lm.generate("I want to say", max_length=30)
falcon_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
falcon_lm.generate("I want to say", max_length=30)
```

Use `generate()` without preprocessing.

```python
prompt = {
    # Token ids for "<|endoftext|> Keras is".
    "token_ids": np.array([[50256, 17337,   292,   318]] * 2),
    # Use `"padding_mask"` to indicate values that should not be overridden.
    "padding_mask": np.array([[1, 1, 1, 1]] * 2),
}
falcon_lm = keras_hub.models.FalconCausalLM.from_preset(
    "falcon_refinedweb_1b_en",
    preprocessor=None,
)
falcon_lm.generate(prompt)
```

Call `fit()` on a single batch.

```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
falcon_lm = keras_hub.models.FalconCausalLM.from_preset("falcon_refinedweb_1b_en")
falcon_lm.fit(x=features, batch_size=2)
```

Call `fit()` without preprocessing.

```python
x = {
    # Token ids for "<|endoftext|> Keras is deep learning library<|endoftext|>"
    "token_ids": np.array([[50256, 17337,   292,   318,  2769,  4673,  5888, 50256, 0]] * 2),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0]] * 2),
}
y = np.array([[17337,   292,   318,  2769,  4673,  5888, 50256, 0, 0]] * 2)
sw = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2)
falcon_lm = keras_hub.models.FalconCausalLM.from_preset(
    "falcon_refinedweb_1b_en",
    preprocessor=None,
)
falcon_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
```

Custom backbone and vocabulary.

```python
vocab = {"<|endoftext|>": 0, "a": 4, "Ä quick": 5, "Ä fox": 6}
merges = ["Ä  q", "u i", "c k", "ui ck", "Ä q uick"]
merges += ["Ä  f", "o x", "Ä f ox"]
tokenizer = keras_hub.models.FalconTokenizer(
    vocabulary=vocab,
    merges=merges,
)
preprocessor = keras_hub.models.FalconCausalLMPreprocessor(
    tokenizer=tokenizer,
    sequence_length=128,
)
backbone = keras_hub.models.FalconBackbone(
    vocabulary_size=50304,
    num_layers=24,
    num_attention_heads=64,
    hidden_dim=2048,
    intermediate_dim=4*2048,
)
falcon_lm = keras_hub.models.FalconCausalLM(
    backbone=backbone,
    preprocessor=preprocessor,
)
falcon_lm.fit(x=features, batch_size=2)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
FalconCausalLM.from_preset(preset, load_weights=True, **kwargs)
```

Instantiate a [`keras_hub.models.Task`]({{< relref "/docs/api/keras_hub/base_classes/task#task-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Task` subclass, you can run `cls.presets.keys()` to list all
built-in presets available on the class.

This constructor can be called in one of two ways. Either from a task
specific base class like `keras_hub.models.CausalLM.from_preset()`, or
from a model class like `keras_hub.models.BertTextClassifier.from_preset()`.
If calling from the a base class, the subclass of the returning object
will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, saved weights will be loaded into
  the model architecture. If `False`, all weights will be
  randomly initialized.

**Examples**

```python
# Load a Gemma generative task.
causal_lm = keras_hub.models.CausalLM.from_preset(
    "gemma_2b_en",
)
# Load a Bert classification task.
model = keras_hub.models.TextClassifier.from_preset(
    "bert_base_en",
    num_classes=2,
)
```

| Preset name             | Parameters | Description                                                                                      |
| ----------------------- | ---------- | ------------------------------------------------------------------------------------------------ |
| falcon_refinedweb_1b_en | 1.31B      | 24-layer Falcon model (Falcon with 1B parameters), trained on 350B tokens of RefinedWeb dataset. |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm.py#L272" >}}

### `generate` method

```python
FalconCausalLM.generate(
    inputs, max_length=None, stop_token_ids="auto", strip_prompt=False
)
```

Generate text given prompt `inputs`.

This method generates text based on given `inputs`. The sampling method
used for generation can be set via the `compile()` method.

If `inputs` are a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), outputs will be generated
"batch-by-batch" and concatenated. Otherwise, all inputs will be handled
as a single batch.

If a `preprocessor` is attached to the model, `inputs` will be
preprocessed inside the `generate()` function and should match the
structure expected by the `preprocessor` layer (usually raw strings).
If a `preprocessor` is not attached, inputs should match the structure
expected by the `backbone`. See the example usage above for a
demonstration of each.

**Arguments**

- **inputs**: python data, tensor data, or a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). If a
  `preprocessor` is attached to the model, `inputs` should match
  the structure expected by the `preprocessor` layer. If a
  `preprocessor` is not attached, `inputs` should match the
  structure expected the `backbone` model.
- **max_length**: Optional. int. The max length of the generated sequence.
  Will default to the max configured `sequence_length` of the
  `preprocessor`. If `preprocessor` is `None`, `inputs` should be
  should be padded to the desired maximum length and this argument
  will be ignored.
- **stop_token_ids**: Optional. `None`, "auto", or tuple of token ids. Defaults
  to "auto" which uses the `preprocessor.tokenizer.end_token_id`.
  Not specifying a processor will produce an error. None stops
  generation after generating `max_length` tokens. You may also
  specify a list of token id's the model should stop on. Note that
  sequences of tokens will each be interpreted as a stop token,
  multi-token stop sequences are not supported.
- **strip_prompt**: Optional. By default, generate() returns the full prompt
  followed by its completion generated by the model. If this option
  is set to True, only the newly generated text is returned.

### `backbone` property

```python
keras_nlp.models.FalconCausalLM.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_nlp.models.FalconCausalLM.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.
