---
title: CausalLMPreprocessor
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm_preprocessor.py#L10" >}}

### `CausalLMPreprocessor` class

```python
keras_nlp.models.CausalLMPreprocessor(
    tokenizer, sequence_length=1024, add_start_token=True, add_end_token=True, **kwargs
)
```

Base class for causal language modeling preprocessing layers.

`CausalLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
create a preprocessing layer for causal language modeling tasks. It is
intended to be paired with a `keras.models.CausalLM` task.

All `CausalLMPreprocessor` take inputs a single input. This can be a single
string or a batch of strings. See examples below. These inputs
will be tokenized and padded/truncated to a fixed sequence length.

This layer will always output a `(x, y, sample_weight)` tuple, where `x`
is a dictionary with the tokenized inputs, `y` contains the tokens from `x`
offset by 1, and `sample_weight` marks where `y` contains padded
values. The exact contents of `x` will vary depending on the model being
used.

a `CausalLMPreprocessor` contains two extra methods, `generate_preprocess`
and `generate_postprocess` for use with generation. See examples below.

All `CausalLMPreprocessor` tasks include a `from_preset()` constructor
which can be used to load a pre-trained config and vocabularies. You can
call the `from_preset()` constructor directly on this base class, in which
case the correct class for you model will be automatically instantiated.

Examples.

```python
preprocessor = keras_hub.models.CausalLMPreprocessor.from_preset(
    "bert_base_en_uncased",
    sequence_length=256, # Optional.
)
# Tokenize, mask and pack a single sentence.
x = "The quick brown fox jumped."
x, y, sample_weight = preprocessor(x)
# Tokenize and pad/truncate a batch of labeled sentences.
x = ["The quick brown fox jumped.", "Call me Ishmael."]
x, y, sample_weight = preprocessor(x)
# With a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
ds = tf.data.Dataset.from_tensor_slices(x)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
# Generate preprocess and postprocess.
x = preprocessor.generate_preprocess(x)  # Tokenized numeric inputs.
x = preprocessor.generate_postprocess(x)  # Detokenized string outputs.
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
CausalLMPreprocessor.from_preset(preset, config_file="preprocessor.json", **kwargs)
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

| Preset name                   | Parameters | Description                                                                                                                                                                                                                                                                   |
| ----------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pali_gemma_3b_mix_224         | 2.92B      | image size 224, mix fine tuned, text sequence length is 256                                                                                                                                                                                                                   |
| pali_gemma_3b_mix_448         | 2.92B      | image size 448, mix fine tuned, text sequence length is 512                                                                                                                                                                                                                   |
| pali_gemma_3b_224             | 2.92B      | image size 224, pre trained, text sequence length is 128                                                                                                                                                                                                                      |
| pali_gemma_3b_448             | 2.92B      | image size 448, pre trained, text sequence length is 512                                                                                                                                                                                                                      |
| pali_gemma_3b_896             | 2.93B      | image size 896, pre trained, text sequence length is 512                                                                                                                                                                                                                      |
| gemma_2b_en                   | 2.51B      | 2 billion parameter, 18-layer, base Gemma model.                                                                                                                                                                                                                              |
| gemma_instruct_2b_en          | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model.                                                                                                                                                                                                                 |
| gemma_1.1_instruct_2b_en      | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                                                                                                                          |
| code_gemma_1.1_2b_en          | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion. The 1.1 update improves model quality.                                                                                                    |
| code_gemma_2b_en              | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                                                                                                                           |
| gemma_7b_en                   | 8.54B      | 7 billion parameter, 28-layer, base Gemma model.                                                                                                                                                                                                                              |
| gemma_instruct_7b_en          | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model.                                                                                                                                                                                                                 |
| gemma_1.1_instruct_7b_en      | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                                                                                                                          |
| code_gemma_7b_en              | 8.54B      | 7 billion parameter, 28-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                                                                                                                           |
| code_gemma_instruct_7b_en     | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code.                                                                                                                                             |
| code_gemma_1.1_instruct_7b_en | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code. The 1.1 update improves model quality.                                                                                                      |
| gemma2_2b_en                  | 2.61B      | 2 billion parameter, 26-layer, base Gemma model.                                                                                                                                                                                                                              |
| gemma2_instruct_2b_en         | 2.61B      | 2 billion parameter, 26-layer, instruction tuned Gemma model.                                                                                                                                                                                                                 |
| gemma2_9b_en                  | 9.24B      | 9 billion parameter, 42-layer, base Gemma model.                                                                                                                                                                                                                              |
| gemma2_instruct_9b_en         | 9.24B      | 9 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                                                                                                                                 |
| gemma2_27b_en                 | 27.23B     | 27 billion parameter, 42-layer, base Gemma model.                                                                                                                                                                                                                             |
| gemma2_instruct_27b_en        | 27.23B     | 27 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                                                                                                                                |
| shieldgemma_2b_en             | 2.61B      | 2 billion parameter, 26-layer, ShieldGemma model.                                                                                                                                                                                                                             |
| shieldgemma_9b_en             | 9.24B      | 9 billion parameter, 42-layer, ShieldGemma model.                                                                                                                                                                                                                             |
| shieldgemma_27b_en            | 27.23B     | 27 billion parameter, 42-layer, ShieldGemma model.                                                                                                                                                                                                                            |
| phi3_mini_4k_instruct_en      | 3.82B      | 3.8 billion parameters, 32 layers, 4k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties.   |
| phi3_mini_128k_instruct_en    | 3.82B      | 3.8 billion parameters, 32 layers, 128k context length, Phi-3 model. The model was trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties. |
| llama2_7b_en                  | 6.74B      | 7 billion parameter, 32-layer, base LLaMA 2 model.                                                                                                                                                                                                                            |
| llama2_7b_en_int8             | 6.74B      | 7 billion parameter, 32-layer, base LLaMA 2 model with activation and weights quantized to int8.                                                                                                                                                                              |
| llama2_instruct_7b_en         | 6.74B      | 7 billion parameter, 32-layer, instruction tuned LLaMA 2 model.                                                                                                                                                                                                               |
| llama2_instruct_7b_en_int8    | 6.74B      | 7 billion parameter, 32-layer, instruction tuned LLaMA 2 model with activation and weights quantized to int8.                                                                                                                                                                 |
| vicuna_1.5_7b_en              | 6.74B      | 7 billion parameter, 32-layer, instruction tuned Vicuna v1.5 model.                                                                                                                                                                                                           |
| llama3_8b_en                  | 8.03B      | 8 billion parameter, 32-layer, base LLaMA 3 model.                                                                                                                                                                                                                            |
| llama3_8b_en_int8             | 8.03B      | 8 billion parameter, 32-layer, base LLaMA 3 model with activation and weights quantized to int8.                                                                                                                                                                              |
| llama3_instruct_8b_en         | 8.03B      | 8 billion parameter, 32-layer, instruction tuned LLaMA 3 model.                                                                                                                                                                                                               |
| llama3_instruct_8b_en_int8    | 8.03B      | 8 billion parameter, 32-layer, instruction tuned LLaMA 3 model with activation and weights quantized to int8.                                                                                                                                                                 |
| mistral_7b_en                 | 7.24B      | Mistral 7B base model                                                                                                                                                                                                                                                         |
| mistral_instruct_7b_en        | 7.24B      | Mistral 7B instruct model                                                                                                                                                                                                                                                     |
| mistral_0.2_instruct_7b_en    | 7.24B      | Mistral 7B instruct Version 0.2 model                                                                                                                                                                                                                                         |
| falcon_refinedweb_1b_en       | 1.31B      | 24-layer Falcon model (Falcon with 1B parameters), trained on 350B tokens of RefinedWeb dataset.                                                                                                                                                                              |
| opt_125m_en                   | 125.24M    | 12-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora.                                                                                                                                                              |
| opt_1.3b_en                   | 1.32B      | 24-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora.                                                                                                                                                              |
| opt_2.7b_en                   | 2.70B      | 32-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora.                                                                                                                                                              |
| opt_6.7b_en                   | 6.70B      | 32-layer OPT model where case in maintained. Trained on BookCorpus, CommonCrawl, Pile, and PushShift.io corpora.                                                                                                                                                              |
| bloom_560m_multi              | 559.21M    | 24-layer Bloom model with hidden dimension of 1024. trained on 45 natural languages and 12 programming languages.                                                                                                                                                             |
| bloom_1.1b_multi              | 1.07B      | 24-layer Bloom model with hidden dimension of 1536. trained on 45 natural languages and 12 programming languages.                                                                                                                                                             |
| bloom_1.7b_multi              | 1.72B      | 24-layer Bloom model with hidden dimension of 2048. trained on 45 natural languages and 12 programming languages.                                                                                                                                                             |
| bloom_3b_multi                | 3.00B      | 30-layer Bloom model with hidden dimension of 2560. trained on 45 natural languages and 12 programming languages.                                                                                                                                                             |
| bloomz_560m_multi             | 559.21M    | 24-layer Bloom model with hidden dimension of 1024. finetuned on crosslingual task mixture (xP3) dataset.                                                                                                                                                                     |
| bloomz_1.1b_multi             | 1.07B      | 24-layer Bloom model with hidden dimension of 1536. finetuned on crosslingual task mixture (xP3) dataset.                                                                                                                                                                     |
| bloomz_1.7b_multi             | 1.72B      | 24-layer Bloom model with hidden dimension of 2048. finetuned on crosslingual task mixture (xP3) dataset.                                                                                                                                                                     |
| bloomz_3b_multi               | 3.00B      | 30-layer Bloom model with hidden dimension of 2560. finetuned on crosslingual task mixture (xP3) dataset.                                                                                                                                                                     |
| gpt2_base_en                  | 124.44M    | 12-layer GPT-2 model where case is maintained. Trained on WebText.                                                                                                                                                                                                            |
| gpt2_medium_en                | 354.82M    | 24-layer GPT-2 model where case is maintained. Trained on WebText.                                                                                                                                                                                                            |
| gpt2_large_en                 | 774.03M    | 36-layer GPT-2 model where case is maintained. Trained on WebText.                                                                                                                                                                                                            |
| gpt2_extra_large_en           | 1.56B      | 48-layer GPT-2 model where case is maintained. Trained on WebText.                                                                                                                                                                                                            |
| gpt2_base_en_cnn_dailymail    | 124.44M    | 12-layer GPT-2 model where case is maintained. Finetuned on the CNN/DailyMail summarization dataset.                                                                                                                                                                          |

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L222" >}}

### `save_to_preset` method

```python
CausalLMPreprocessor.save_to_preset(preset_dir)
```

Save preprocessor to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

### `tokenizer` property

```python
keras_nlp.models.CausalLMPreprocessor.tokenizer
```

The tokenizer used to tokenize strings.
