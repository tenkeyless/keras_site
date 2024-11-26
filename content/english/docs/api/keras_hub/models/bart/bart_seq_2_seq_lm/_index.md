---
title: BartSeq2SeqLM model
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/bart/bart_seq_2_seq_lm.py#L12" >}}

### `BartSeq2SeqLM` class

```python
keras_hub.models.BartSeq2SeqLM(backbone, preprocessor=None, **kwargs)
```

An end-to-end BART model for seq2seq language modeling.

A seq2seq language model (LM) is an encoder-decoder model which is used for
conditional text generation. The encoder is given a "context" text (fed to
the encoder), and the decoder predicts the next token based on both the
encoder inputs and the previous tokens. You can finetune `BartSeq2SeqLM` to
generate text for any seq2seq task (e.g., translation or summarization).

This model has a `generate()` method, which generates text based on
encoder inputs and an optional prompt for the decoder. The generation
strategy used is controlled by an additional `sampler` argument passed to
`compile()`. You can recompile the model with different `keras_hub.samplers`
objects to control the generation. By default, `"top_k"` sampling will be
used.

This model can optionally be configured with a `preprocessor` layer, in
which case it will automatically apply preprocessing to string inputs during
`fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
when creating the model with `from_preset()`.

Disclaimer: Pre-trained models are provided on an "as is" basis, without
warranties or conditions of any kind. The underlying model is provided by a
third party and subject to a separate license, available
[here](https://github.com/facebookresearch/fairseq/).

**Arguments**

- **backbone**: A `keras_hub.models.BartBackbone` instance.
- **preprocessor**: A [`keras_hub.models.BartSeq2SeqLMPreprocessor`]({{< relref "/docs/api/keras_hub/models/bart/bart_seq_2_seq_lm_preprocessor#bartseq2seqlmpreprocessor-class" >}}) or `None`.
  If `None`, this model will not apply preprocessing, and inputs
  should be preprocessed before calling the model.

**Examples**

Use `generate()` to do text generation, given an input context.

```python
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
bart_lm.generate("The quick brown fox", max_length=30)
# Generate with batched inputs.
bart_lm.generate(["The quick brown fox", "The whale"], max_length=30)
```

Compile the `generate()` function with a custom sampler.

```python
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
bart_lm.compile(sampler="greedy")
bart_lm.generate("The quick brown fox", max_length=30)
```

Use `generate()` with encoder inputs and an incomplete decoder input (prompt).

```python
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
bart_lm.generate(
    {
        "encoder_text": "The quick brown fox",
        "decoder_text": "The fast"
    }
)
```

Use `generate()` without preprocessing.

```python
# Preprocessed inputs, with encoder inputs corresponding to
# "The quick brown fox", and the decoder inputs to "The fast". Use
# `"padding_mask"` to indicate values that should not be overridden.
prompt = {
    "encoder_token_ids": np.array([[0, 133, 2119, 6219, 23602, 2, 1, 1]]),
    "encoder_padding_mask": np.array(
        [[True, True, True, True, True, True, False, False]]
    ),
    "decoder_token_ids": np.array([[2, 0, 133, 1769, 2, 1, 1]]),
    "decoder_padding_mask": np.array([[True, True, True, True, False, False]])
}
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
    "bart_base_en",
    preprocessor=None,
)
bart_lm.generate(prompt)
```

Call `fit()` on a single batch.

```python
features = {
    "encoder_text": ["The quick brown fox jumped.", "I forgot my homework."],
    "decoder_text": ["The fast hazel fox leapt.", "I forgot my assignment."]
}
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
bart_lm.fit(x=features, batch_size=2)
```

Call `fit()` without preprocessing.

```python
x = {
    "encoder_token_ids": np.array([[0, 133, 2119, 2, 1]] * 2),
    "encoder_padding_mask": np.array([[1, 1, 1, 1, 0]] * 2),
    "decoder_token_ids": np.array([[2, 0, 133, 1769, 2]] * 2),
    "decoder_padding_mask": np.array([[1, 1, 1, 1, 1]] * 2),
}
y = np.array([[0, 133, 1769, 2, 1]] * 2)
sw = np.array([[1, 1, 1, 1, 0]] * 2)
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
    "bart_base_en",
    preprocessor=None,
)
bart_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
```

Custom backbone and vocabulary.

```python
features = {
    "encoder_text": [" afternoon sun"],
    "decoder_text": ["noon sun"],
}
vocab = {
    "<s>": 0,
    "<pad>": 1,
    "</s>": 2,
    "Ä after": 5,
    "noon": 6,
    "Ä sun": 7,
}
merges = ["Ä  a", "Ä  s", "Ä  n", "e r", "n o", "o n", "Ä s u", "Ä a f", "no on"]
merges += ["Ä su n", "Ä af t", "Ä aft er"]
tokenizer = keras_hub.models.BartTokenizer(
    vocabulary=vocab,
    merges=merges,
)
preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor(
    tokenizer=tokenizer,
    encoder_sequence_length=128,
    decoder_sequence_length=128,
)
backbone = keras_hub.models.BartBackbone(
    vocabulary_size=50265,
    num_layers=6,
    num_heads=12,
    hidden_dim=768,
    intermediate_dim=3072,
    max_sequence_length=128,
)
bart_lm = keras_hub.models.BartSeq2SeqLM(
    backbone=backbone,
    preprocessor=preprocessor,
)
bart_lm.fit(x=features, batch_size=2)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
BartSeq2SeqLM.from_preset(preset, load_weights=True, **kwargs)
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

| Preset name       | Parameters | Description                                                                                             |
| ----------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| bart_base_en      | 139.42M    | 6-layer BART model where case is maintained. Trained on BookCorpus, English Wikipedia and CommonCrawl.  |
| bart_large_en     | 406.29M    | 12-layer BART model where case is maintained. Trained on BookCorpus, English Wikipedia and CommonCrawl. |
| bart_large_en_cnn | 406.29M    | The `bart_large_en` backbone model fine-tuned on the CNN+DM summarization dataset.                      |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm.py#L272" >}}

### `generate` method

```python
BartSeq2SeqLM.generate(
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
keras_hub.models.BartSeq2SeqLM.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_hub.models.BartSeq2SeqLM.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.
