---
title: BartSeq2SeqLMPreprocessor layer
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/bart/bart_seq_2_seq_lm_preprocessor.py#L8" >}}

### `BartSeq2SeqLMPreprocessor` class

`keras_hub.models.BartSeq2SeqLMPreprocessor(     tokenizer, encoder_sequence_length=1024, decoder_sequence_length=1024, **kwargs )`

BART Seq2Seq LM preprocessor.

This layer is used as preprocessor for seq2seq tasks using the BART model. This class subclasses `keras_hub.models.BartPreprocessor` and keeps most of its functionality. It has two changes from the superclass:

1.  Sets the `y` (label) and `sample_weights` fields by shifting the decoder input sequence one step towards the left. Both these fields are inferred internally, and any passed values will be ignored.
2.  Drops the last token from the decoder input sequence as it does not have a successor.

**Arguments**

- **tokenizer**: A `keras_hub.models.BartTokenizer` instance.
- **encoder_sequence_length**: The length of the packed encoder inputs.
- **decoder_sequence_length**: The length of the packed decoder inputs.

**Call arguments**

- **x**: A dictionary with `encoder_text` and `decoder_text` as its keys. Each value in the dictionary should be a tensor of single string sequences. Inputs may be batched or unbatched. Raw python inputs will be converted to tensors.
- **y**: Label data. Should always be `None` as the layer generates labels by shifting the decoder input sequence one step to the left.
- **sample_weight**: Label weights. Should always be `None` as the layer generates label weights by shifting the padding mask one step to the left.

**Examples**

Directly calling the layer on data

`preprocessor = keras_hub.models.BartPreprocessor.from_preset("bart_base_en")  # Preprocess unbatched inputs. inputs = {     "encoder_text": "The fox was sleeping.",     "decoder_text": "The fox was awake." } preprocessor(inputs)  # Preprocess batched inputs. inputs = {     "encoder_text": ["The fox was sleeping.", "The lion was quiet."],     "decoder_text": ["The fox was awake.", "The lion was roaring."] } preprocessor(inputs)  # Custom vocabulary. vocab = {     "<s>": 0,     "<pad>": 1,     "</s>": 2,     "Ġafter": 5,     "noon": 6,     "Ġsun": 7, } merges = ["Ġ a", "Ġ s", "Ġ n", "e r", "n o", "o n", "Ġs u", "Ġa f", "no on"] merges += ["Ġsu n", "Ġaf t", "Ġaft er"]  tokenizer = keras_hub.models.BartTokenizer(     vocabulary=vocab,     merges=merges, ) preprocessor = keras_hub.models.BartPreprocessor(     tokenizer=tokenizer,     encoder_sequence_length=20,     decoder_sequence_length=10, ) inputs = {     "encoder_text": "The fox was sleeping.",     "decoder_text": "The fox was awake." } preprocessor(inputs)`

Mapping with [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

`preprocessor = keras_hub.models.BartPreprocessor.from_preset("bart_base_en")  # Map single sentences. features = {     "encoder_text": tf.constant(         ["The fox was sleeping.", "The lion was quiet."]     ),     "decoder_text": tf.constant(         ["The fox was awake.", "The lion was roaring."]     ) } ds = tf.data.Dataset.from_tensor_slices(features) ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)`

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

`BartSeq2SeqLMPreprocessor.from_preset(     preset, config_file="preprocessor.json", **kwargs )`

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

bart_base_en

139.42M

6-layer BART model where case is maintained. Trained on BookCorpus, English Wikipedia and CommonCrawl.

bart_large_en

406.29M

12-layer BART model where case is maintained. Trained on BookCorpus, English Wikipedia and CommonCrawl.

bart_large_en_cnn

406.29M

The `bart_large_en` backbone model fine-tuned on the CNN+DM summarization dataset.

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/seq_2_seq_lm_preprocessor.py#L144" >}}

### `generate_preprocess` method

`BartSeq2SeqLMPreprocessor.generate_preprocess(     x, encoder_sequence_length=None, decoder_sequence_length=None, sequence_length=None )`

Convert encoder and decoder input strings to integer token inputs for generation.

Similar to calling the layer for training, this method takes in a dict containing `"encoder_text"` and `"decoder_text"`, with strings or tensor strings for values, tokenizes and packs the input, and computes a padding mask masking all inputs not filled in with a padded value.

Unlike calling the layer for training, this method does not compute labels and will never append a tokenizer.end_token_id to the end of the decoder sequence (as generation is expected to continue at the end of the inputted decoder prompt).

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/seq_2_seq_lm_preprocessor.py#L205" >}}

### `generate_postprocess` method

`BartSeq2SeqLMPreprocessor.generate_postprocess(x)`

Convert integer token output to strings for generation.

This method reverses `generate_preprocess()`, by first removing all padding and start/end tokens, and then converting the integer sequence back to a string.

---

### `tokenizer` property

`keras_hub.models.BartSeq2SeqLMPreprocessor.tokenizer`

The tokenizer used to tokenize strings.

---
