---
title: GemmaCausalLM model
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/gemma/gemma_causal_lm.py#L13" >}}

### `GemmaCausalLM` class

```python
keras_hub.models.GemmaCausalLM(backbone, preprocessor=None, **kwargs)
```

An end-to-end Gemma model for causal language modeling.

A causal language model (LM) predicts the next token based on previous
tokens. This task setup can be used to train the model unsupervised on
plain text input, or to autoregressively generate plain text similar to
the data used for training. This task can be used for pre-training or
fine-tuning a Gemma model, simply by calling `fit()`.

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

- **backbone**: A [`keras_hub.models.GemmaBackbone`]({{< relref "/docs/api/keras_hub/models/gemma/gemma_backbone#gemmabackbone-class" >}}) instance.
- **preprocessor**: A [`keras_hub.models.GemmaCausalLMPreprocessor`]({{< relref "/docs/api/keras_hub/models/gemma/gemma_causal_lm_preprocessor#gemmacausallmpreprocessor-class" >}}) or `None`.
  If `None`, this model will not apply preprocessing, and inputs
  should be preprocessed before calling the model.

**Examples**

Use `generate()` to do text generation.

```python
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.generate("I want to say", max_length=30)
# Generate with batched prompts.
gemma_lm.generate(["This is a", "Where are you"], max_length=30)
```

Compile the `generate()` function with a custom sampler.

```python
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.compile(sampler="top_k")
gemma_lm.generate("I want to say", max_length=30)
gemma_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
gemma_lm.generate("I want to say", max_length=30)
```

Use `generate()` without preprocessing.

```python
prompt = {
    # Token ids for "<bos> Keras is".
    "token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
    # Use `"padding_mask"` to indicate values that should not be overridden.
    "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
}
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_2b_en",
    preprocessor=None,
)
gemma_lm.generate(prompt)
```

Call `fit()` on a single batch.

```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.fit(x=features, batch_size=2)
```

Call `fit()` with LoRA fine-tuning enabled.

```python
features = ["The quick brown fox jumped.", "I forgot my homework."]
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma.backbone.enable_lora(rank=4)
gemma_lm.fit(x=features, batch_size=2)
```

Call `fit()` without preprocessing.

```python
x = {
    # Token ids for "<bos> Keras is deep learning library<eos>"
    "token_ids": np.array([[2, 214064, 603, 5271, 6044, 9581, 1, 0]] * 2),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
}
y = np.array([[214064, 603, 5271, 6044, 9581, 3, 0, 0]] * 2)
sw = np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_2b_en",
    preprocessor=None,
)
gemma_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
```

Custom backbone and vocabulary.

```python
tokenizer = keras_hub.models.GemmaTokenizer(
    proto="proto.spm",
)
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor(
    tokenizer=tokenizer,
    sequence_length=128,
)
backbone = keras_hub.models.GemmaBackbone(
    vocabulary_size=30552,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    intermediate_dim=512,
    max_sequence_length=128,
)
gemma_lm = keras_hub.models.GemmaCausalLM(
    backbone=backbone,
    preprocessor=preprocessor,
)
gemma_lm.fit(x=features, batch_size=2)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
GemmaCausalLM.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name                   | Parameters | Description                                                                                                                                                                |
| ----------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| gemma_2b_en                   | 2.51B      | 2 billion parameter, 18-layer, base Gemma model.                                                                                                                           |
| gemma_instruct_2b_en          | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model.                                                                                                              |
| gemma_1.1_instruct_2b_en      | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                       |
| code_gemma_1.1_2b_en          | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion. The 1.1 update improves model quality. |
| code_gemma_2b_en              | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                        |
| gemma_7b_en                   | 8.54B      | 7 billion parameter, 28-layer, base Gemma model.                                                                                                                           |
| gemma_instruct_7b_en          | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model.                                                                                                              |
| gemma_1.1_instruct_7b_en      | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                       |
| code_gemma_7b_en              | 8.54B      | 7 billion parameter, 28-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                        |
| code_gemma_instruct_7b_en     | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code.                                          |
| code_gemma_1.1_instruct_7b_en | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code. The 1.1 update improves model quality.   |
| gemma2_2b_en                  | 2.61B      | 2 billion parameter, 26-layer, base Gemma model.                                                                                                                           |
| gemma2_instruct_2b_en         | 2.61B      | 2 billion parameter, 26-layer, instruction tuned Gemma model.                                                                                                              |
| gemma2_9b_en                  | 9.24B      | 9 billion parameter, 42-layer, base Gemma model.                                                                                                                           |
| gemma2_instruct_9b_en         | 9.24B      | 9 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                              |
| gemma2_27b_en                 | 27.23B     | 27 billion parameter, 42-layer, base Gemma model.                                                                                                                          |
| gemma2_instruct_27b_en        | 27.23B     | 27 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                             |
| shieldgemma_2b_en             | 2.61B      | 2 billion parameter, 26-layer, ShieldGemma model.                                                                                                                          |
| shieldgemma_9b_en             | 9.24B      | 9 billion parameter, 42-layer, ShieldGemma model.                                                                                                                          |
| shieldgemma_27b_en            | 27.23B     | 27 billion parameter, 42-layer, ShieldGemma model.                                                                                                                         |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm.py#L272" >}}

### `generate` method

```python
GemmaCausalLM.generate(
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
keras_hub.models.GemmaCausalLM.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_hub.models.GemmaCausalLM.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/gemma/gemma_causal_lm.py#L305" >}}

### `score` method

```python
GemmaCausalLM.score(
    token_ids,
    padding_mask=None,
    scoring_mode="logits",
    layer_intercept_fn=None,
    target_ids=None,
)
```

Score a generation represented by the provided token ids.

**Arguments**

- **token_ids**: A [batch\_size, num\_tokens] tensor containing tokens
  to score. Typically, this tensor captures the output from a call
  to `GemmaCausalLM.generate()`, i.e., tokens for both the input
  text and the model-generated text.
- **padding_mask**: A [batch\_size, num\_tokens] tensor indicating the
  tokens that should be preserved during generation. This is an
  artifact required by the GemmaBackbone and isn't influential on
  the computation of this function. If omitted, this function uses
  `keras.ops.ones()` to create a tensor of the appropriate shape.
- **scoring_mode**: The type of scores to return, either "logits" or
  "loss", both will be per input token.
- **layer_intercept_fn**: An optional function for augmenting activations
  with additional computation, for example, as part of
  interpretability research. This function will be passed the
  activations as its first parameter and a numeric index
  associated with that backbone layer. _This index \_is not_ an
  index into `self.backbone.layers`\_. The index -1 accompanies the
  embeddings returned by calling `self.backbone.token_embedding()`
  on `token_ids` in the forward direction. All subsequent indexes
  will be 0-based indices for the activations returned by each of
  the Transformers layers in the backbone. This function must
  return a [batch\_size, num\_tokens, hidden\_dims] tensor
  that can be passed as an input to the next layer in the model.
- **target_ids**: An [batch\_size, num\_tokens] tensor containing the
  predicted tokens against which the loss should be computed. If a
  span of tokens is provided (sequential truthy values along
  axis=1 in the tensor), the loss will be computed as the
  aggregate across those tokens.

**Raises**

- **ValueError**: If an unsupported scoring_mode is provided, or if the
  target_ids are not provided when using ScoringMode.LOSS.

**Returns**

The per-token scores as a tensor of size
[batch\_size, num\_tokens, vocab\_size] in "logits" mode, or
[batch\_size, num\_tokens] in "loss" mode.

**Example**

Compute gradients between embeddings and loss scores with TensorFlow:

```python
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_2b_en"
)
generations = gemma_lm.generate(
    ["This is a", "Where are you"],
    max_length=30
)
preprocessed = gemma_lm.preprocessor.generate_preprocess(generations)
generation_ids = preprocessed["token_ids"]
padding_mask = preprocessed["padding_mask"]
target_ids = keras.ops.roll(generation_ids, shift=-1, axis=1)
embeddings = None
with tf.GradientTape(watch_accessed_variables=True) as tape:
    def layer_intercept_fn(x, i):
        if i == -1:
            nonlocal embeddings, tape
            embeddings = x
            tape.watch(embeddings)
        return x
    losses = gemma_lm.score(
        token_ids=generation_ids,
        padding_mask=padding_mask,
        scoring_mode="loss",
        layer_intercept_fn=layer_intercept_fn,
        target_ids=target_ids,
    )
grads = tape.gradient(losses, embeddings)
```
