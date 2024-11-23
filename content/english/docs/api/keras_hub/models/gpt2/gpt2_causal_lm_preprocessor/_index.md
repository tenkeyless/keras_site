---
title: GPT2CausalLMPreprocessor layer
toc: false
---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/gpt2/gpt2_causal_lm_preprocessor.py#L7)

### `GPT2CausalLMPreprocessor` class

`keras_hub.models.GPT2CausalLMPreprocessor(     tokenizer, sequence_length=1024, add_start_token=True, add_end_token=True, **kwargs )`

GPT2 Causal LM preprocessor.

This preprocessing layer is meant for use with [`keras_hub.models.GPT2CausalLM`](/api/keras_hub/models/gpt2/gpt2_causal_lm#gpt2causallm-class). By default, it will take in batches of strings, and return outputs in a `(x, y, sample_weight)` format, where the `y` label is the next token id in the `x` sequence.

For use with generation, the layer also exposes two methods `generate_preprocess()` and `generate_postprocess()`. When this preprocessor is attached to a [`keras_hub.models.GPT2CausalLM`](/api/keras_hub/models/gpt2/gpt2_causal_lm#gpt2causallm-class) instance, these methods will be called implicitly in `generate()`. They can also be called standalone (e.g. to precompute preprocessing inputs for generation in a separate process).

**Arguments**

- **tokenizer**: A `keras_hub.models.GPT2Tokenizer` instance.
- **sequence_length**: The length of the packed inputs.
- **add_start_token**: If `True`, the preprocessor will prepend the tokenizer start token to each input sequence.
- **add_end_token**: If `True`, the preprocessor will append the tokenizer end token to each input sequence.

**Call arguments**

- **x**: A string, [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) or list of python strings.
- **y**: Label data. Should always be `None` as the layer generates labels.
- **sample_weight**: Label weights. Should always be `None` as the layer generates label weights.
- **sequence_length**: Pass to override the configured `sequence_length` of the layer.

**Examples**

`# Load the preprocessor from a preset. preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(     "gpt2_base_en" )  # Tokenize and pack a single sentence. sentence = tf.constant("League of legends") preprocessor(sentence) # Same output. preprocessor("League of legends")  # Tokenize a batch of sentences. sentences = tf.constant(["Taco tuesday", "Fish taco please!"]) preprocessor(sentences) # Same output. preprocessor(["Taco tuesday", "Fish taco please!"])  # Map a dataset to preprocess a single sentence. features = tf.constant(     [         "Avatar 2 is amazing!",         "Well, I am not sure.",     ] ) labels = tf.constant([1, 0]) ds = tf.data.Dataset.from_tensor_slices((features, labels)) ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)  # Map a dataset to preprocess unlabled sentences. ds = tf.data.Dataset.from_tensor_slices(features) ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)`

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132)

### `from_preset` method

`GPT2CausalLMPreprocessor.from_preset(     preset, config_file="preprocessor.json", **kwargs )`

Instantiate a [`keras_hub.models.Preprocessor`](/api/keras_hub/base_classes/preprocessor#preprocessor-class) from a model preset.

A preset is a directory of configs, weights and other file assets used to save and load a pre-trained model. The `preset` can be passed as one of:

1.  a built-in preset identifier like `'bert_base_en'`
2.  a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3.  a Hugging Face handle like `'hf://user/bert_base_en'`
4.  a path to a local preset directory like `'./bert_base_en'`

For any `Preprocessor` subclass, you can run `cls.presets.keys()` to list all built-in presets available on the class.

As there are usually multiple preprocessing classes for a given model, this method should be called on a specific subclass like `keras_hub.models.BertTextClassifierPreprocessor.from_preset()`.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models handle, a Hugging Face handle, or a path to a local directory.

**Examples**

`# Load a preprocessor for Gemma generation. preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(     "gemma_2b_en", )  # Load a preprocessor for Bert classification. preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(     "bert_base_en", )`

Preset name

Parameters

Description

gpt2_base_en

124.44M

12-layer GPT-2 model where case is maintained. Trained on WebText.

gpt2_medium_en

354.82M

24-layer GPT-2 model where case is maintained. Trained on WebText.

gpt2_large_en

774.03M

36-layer GPT-2 model where case is maintained. Trained on WebText.

gpt2_extra_large_en

1.56B

48-layer GPT-2 model where case is maintained. Trained on WebText.

gpt2_base_en_cnn_dailymail

124.44M

12-layer GPT-2 model where case is maintained. Finetuned on the CNN/DailyMail summarization dataset.

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm_preprocessor.py#L114)

### `generate_preprocess` method

`GPT2CausalLMPreprocessor.generate_preprocess(x, sequence_length=None)`

Convert strings to integer token input for generation.

Similar to calling the layer for training, this method takes in strings or tensor strings, tokenizes and packs the input, and computes a padding mask masking all inputs not filled in with a padded value.

Unlike calling the layer for training, this method does not compute labels and will never append a `tokenizer.end_token_id` to the end of the sequence (as generation is expected to continue at the end of the inputted prompt).

---

[\[source\]](https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/causal_lm_preprocessor.py#L143)

### `generate_postprocess` method

`GPT2CausalLMPreprocessor.generate_postprocess(x)`

Convert integer token output to strings for generation.

This method reverses `generate_preprocess()`, by first removing all padding and start/end tokens, and then converting the integer sequence back to a string.

---

### `tokenizer` property

`keras_hub.models.GPT2CausalLMPreprocessor.tokenizer`

The tokenizer used to tokenize strings.

---
