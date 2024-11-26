---
title: Phi3CausalLM model
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/phi3/phi3_causal_lm.py#L12" >}}

### `Phi3CausalLM` class

```python
keras_nlp.models.Phi3CausalLM(backbone, preprocessor=None, **kwargs)
```

An end-to-end Phi3 model for causal language modeling.

A causal language model (LM) predicts the next token based on previous
tokens. This task setup can be used to train the model unsupervised on
plain text input, or to autoregressively generate plain text similar to
the data used for training. This task can be used for pre-training or
fine-tuning a Phi-3 model, simply by calling `fit()`.

This model has a `generate()` method, which generates text based on a
prompt. The generation strategy used is controlled by an additional
`sampler` argument on `compile()`. You can recompile the model with
different `keras_hub.samplers` objects to control the generation. By
default, `"top_k"` sampling will be used.

**Arguments**

- **backbone**: A [`keras_hub.models.Phi3Backbone`]({{< relref "/docs/api/keras_hub/models/phi3/phi3_backbone#phi3backbone-class" >}}) instance.
- **preprocessor**: A [`keras_hub.models.Phi3CausalLMPreprocessor`]({{< relref "/docs/api/keras_hub/models/phi3/phi3_causal_lm_preprocessor#phi3causallmpreprocessor-class" >}}) or `None`.
  If `None`, this model will not apply preprocessing, and inputs
  should be preprocessed before calling the model.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
Phi3CausalLM.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name                | Parameters | Description                                                                                                                                                                                                                                                                   |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| phi3_mini_4k_instruct_en   | 3.82B      | 3.8 billion parameters, 32 layers, 4k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties.   |
| phi3_mini_128k_instruct_en | 3.82B      | 3.8 billion parameters, 32 layers, 128k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties. |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/phi3/phi3_causal_lm.py#L190" >}}

### `generate` method

```python
Phi3CausalLM.generate(inputs, max_length=None, stop_token_ids="auto")
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
keras_nlp.models.Phi3CausalLM.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_nlp.models.Phi3CausalLM.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.