---
title: OPTCausalLMPreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/opt/opt_causal_lm_preprocessor.py#L7" >}}

### `OPTCausalLMPreprocessor` class

```python
keras_hub.models.OPTCausalLMPreprocessor(
    tokenizer, sequence_length=1024, add_start_token=True, add_end_token=True, **kwargs
)
```

OPT Causal LM preprocessor.

This preprocessing layer is primarily meant to be used with
[`keras_hub.models.OPTCausalLM`]({{< relref "/docs/api/keras_hub/models/opt/opt_causal_lm#optcausallm-class" >}}). By default, it will take in batches of
strings, and return outputs in a `(x, y, sample_weight)` format, where the
`y` label is the next token id in the `x` sequence. For use with generation,
pass `return_labels=False` in which case the output will simply be the
encoded string features.

**Arguments**

- **tokenizer**: A `keras_hub.models.OPTTokenizer` instance.
- **sequence_length**: The length of the packed inputs.
- **add_start_token**: If `True`, the preprocessor will prepend the tokenizer
  start token to each input sequence.
- **add_end_token**: If `True`, the preprocessor will append the tokenizer
  end token to each input sequence.

**Call arguments**

- **x**: A string, [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) or list of python strings.
- **y**: Label data. Should always be `None` as the layer generates labels.
- **sample_weight**: Label weights. Should always be `None` as the layer
  generates label weights.
- **sequence_length**: Pass to override the configured `sequence_length` of
  the layer.
- **add_start_token**: Pass to override the configured value of
  `add_start_token` on the layer.
- **add_end_token**: Pass to override the configured value of
  `add_end_token` on the layer.
- **return_labels**: If `True`, the output `"token_ids"` will be offset by one
  and returned as labels. If `False` only features will be returned.

**Examples**

```python
# Load the preprocessor from a preset.
preprocessor = keras_hub.models.OPTCausalLMPreprocessor.from_preset(
    "opt_125m_en"
)
# Tokenize and pack a single sentence.
sentence = tf.constant("League of legends")
preprocessor(sentence)
# Same output.
preprocessor("League of legends")
# Tokenize a batch of sentences.
sentences = tf.constant(["Taco tuesday", "Fish taco please!"])
preprocessor(sentences)
# Same output.
preprocessor(["Taco tuesday", "Fish taco please!"])
# Map a dataset to preprocess a single sentence.
features = tf.constant(
    [
        "Avatar 2 is amazing!",
        "Well, I am not sure.",
    ]
)
labels = tf.constant([1, 0])
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Map a dataset to preprocess unlabled sentences.
ds = tf.data.Dataset.from_tensor_slices(features)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
OPTCausalLMPreprocessor.from_preset(
    preset, config_file="preprocessor.json", **kwargs
)
```

Instantiate a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Preprocessor` subclass, you can run `cls.presets.keys()` to
list all built-in presets available on the class.

As there are usually multiple preprocessing classes for a given model,
this method should be called on a specific subclass like
`keras_hub.models.BertTextClassifierPreprocessor.from_preset()`.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.

**Examples**

```python
# Load a preprocessor for Gemma generation.
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en",
)
# Load a preprocessor for Bert classification.
preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en",
)
```

| Preset name | Parameters | Description                                                                                                      |
| ----------- | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| opt_125m_en | 125.24M    | 12-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |
| opt_1.3b_en | 1.32B      | 24-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |
| opt_2.7b_en | 2.70B      | 32-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |
| opt_6.7b_en | 6.70B      | 32-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora. |

### `tokenizer` property

```python
keras_hub.models.OPTCausalLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
