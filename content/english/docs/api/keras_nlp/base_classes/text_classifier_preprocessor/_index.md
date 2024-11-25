---
title: TextClassifierPreprocessor
toc: true
weight: 9
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/text_classifier_preprocessor.py#L11" >}}

### `TextClassifierPreprocessor` class

`keras_hub.models.TextClassifierPreprocessor(     tokenizer, sequence_length=512, truncate="round_robin", **kwargs )`

Base class for text classification preprocessing layers.

`TextClassifierPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to create a preprocessing layer for text classification tasks. It is intended to be paired with a [`keras_hub.models.TextClassifier`](/api/keras_hub/base_classes/text_classifier#textclassifier-class) task.

All `TextClassifierPreprocessor` take inputs three ordered inputs, `x`, `y`, and `sample_weight`. `x`, the first input, should always be included. It can be a single string, a batch of strings, or a tuple of batches of string segments that should be combined into a single sequence. See examples below. `y` and `sample_weight` are optional inputs that will be passed through unaltered. Usually, `y` will be the classification label, and `sample_weight` will not be provided.

The layer will output either `x`, an `(x, y)` tuple if labels were provided, or an `(x, y, sample_weight)` tuple if labels and sample weight were provided. `x` will be a dictionary with tokenized input, the exact contents of the dictionary will depend on the model being used.

All `TextClassifierPreprocessor` tasks include a `from_preset()` constructor which can be used to load a pre-trained config and vocabularies. You can call the `from_preset()` constructor directly on this base class, in which case the correct class for you model will be automatically instantiated.

Examples.

`` preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(     "bert_base_en_uncased",     sequence_length=256, # Optional. )  # Tokenize and pad/truncate a single sentence. x = "The quick brown fox jumped." x = preprocessor(x)  # Tokenize and pad/truncate a labeled sentence. x, y = "The quick brown fox jumped.", 1 x, y = preprocessor(x, y)  # Tokenize and pad/truncate a batch of labeled sentences. x, y = ["The quick brown fox jumped.", "Call me Ishmael."], [1, 0] x, y = preprocessor(x, y)  # Tokenize and combine a batch of labeled sentence pairs. first = ["The quick brown fox jumped.", "Call me Ishmael."] second = ["The fox tripped.", "Oh look, a whale."] labels = [1, 0] x, y = (first, second), labels x, y = preprocessor(x, y)  # Use a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). ds = tf.data.Dataset.from_tensor_slices(((first, second), labels)) ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE) ``

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

`TextClassifierPreprocessor.from_preset(     preset, config_file="preprocessor.json", **kwargs )`

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

roberta_base_en

124.05M

12-layer RoBERTa model where case is maintained.Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText.

roberta_large_en

354.31M

24-layer RoBERTa model where case is maintained.Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText.

xlm_roberta_base_multi

277.45M

12-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages.

xlm_roberta_large_multi

558.84M

24-layer XLM-RoBERTa model where case is maintained. Trained on CommonCrawl in 100 languages.

bert_tiny_en_uncased

4.39M

2-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

bert_small_en_uncased

28.76M

4-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

bert_medium_en_uncased

41.37M

8-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

bert_base_en_uncased

109.48M

12-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

bert_base_en

108.31M

12-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus.

bert_base_zh

102.27M

12-layer BERT model. Trained on Chinese Wikipedia.

bert_base_multi

177.85M

12-layer BERT model where case is maintained. Trained on trained on Wikipedias of 104 languages

bert_large_en_uncased

335.14M

24-layer BERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

bert_large_en

333.58M

24-layer BERT model where case is maintained. Trained on English Wikipedia + BooksCorpus.

bert_tiny_en_uncased_sst2

4.39M

The bert_tiny_en_uncased backbone model fine-tuned on the SST-2 sentiment analysis dataset.

albert_base_en_uncased

11.68M

12-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

albert_large_en_uncased

17.68M

24-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

albert_extra_large_en_uncased

58.72M

24-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

albert_extra_extra_large_en_uncased

222.60M

12-layer ALBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus.

deberta_v3_extra_small_en

70.68M

12-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText.

deberta_v3_small_en

141.30M

6-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText.

deberta_v3_base_en

183.83M

12-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText.

deberta_v3_large_en

434.01M

24-layer DeBERTaV3 model where case is maintained. Trained on English Wikipedia, BookCorpus and OpenWebText.

deberta_v3_base_multi

278.22M

12-layer DeBERTaV3 model where case is maintained. Trained on the 2.5TB multilingual CC100 dataset.

distil_bert_base_en_uncased

66.36M

6-layer DistilBERT model where all input is lowercased. Trained on English Wikipedia + BooksCorpus using BERT as the teacher model.

distil_bert_base_en

65.19M

6-layer DistilBERT model where case is maintained. Trained on English Wikipedia + BooksCorpus using BERT as the teacher model.

distil_bert_base_multi

134.73M

6-layer DistilBERT model where case is maintained. Trained on Wikipedias of 104 languages

f_net_base_en

82.86M

12-layer FNet model where case is maintained. Trained on the C4 dataset.

f_net_large_en

236.95M

24-layer FNet model where case is maintained. Trained on the C4 dataset.

---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L222" >}}

### `save_to_preset` method

`TextClassifierPreprocessor.save_to_preset(preset_dir)`

Save preprocessor to a preset directory.

**Arguments**

- **preset_dir**: The path to the local model preset directory.

---

### `tokenizer` property

`keras_hub.models.TextClassifierPreprocessor.tokenizer`

The tokenizer used to tokenize strings.

---
