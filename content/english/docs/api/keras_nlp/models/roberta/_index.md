---
title: RoBERTa
toc: true
weight: 18
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

Models, tokenizers, and preprocessing layers for RoBERTa,
as described in ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692).

For a full list of available **presets**, see the
[models page]({{< relref "/docs/api/keras_nlp/models" >}}).

### [RobertaTokenizer]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_tokenizer/" >}})

- [RobertaTokenizer class]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_tokenizer/#robertatokenizer-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_tokenizer/#from_preset-method" >}})

### [RobertaBackbone model]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_backbone/" >}})

- [RobertaBackbone class]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_backbone/#robertabackbone-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_backbone/#from_preset-method" >}})
- [token\_embedding property]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_backbone/#token_embedding-property" >}})

### [RobertaTextClassifier model]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier/" >}})

- [RobertaTextClassifier class]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier/#robertatextclassifier-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier/#from_preset-method" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier/#backbone-property" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier/#preprocessor-property" >}})

### [RobertaTextClassifierPreprocessor layer]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier_preprocessor/" >}})

- [RobertaTextClassifierPreprocessor class]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier_preprocessor/#robertatextclassifierpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier_preprocessor/#from_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_text_classifier_preprocessor/#tokenizer-property" >}})

### [RobertaMaskedLM model]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm/" >}})

- [RobertaMaskedLM class]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm/#robertamaskedlm-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm/#from_preset-method" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm/#backbone-property" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm/#preprocessor-property" >}})

### [RobertaMaskedLMPreprocessor layer]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm_preprocessor/" >}})

- [RobertaMaskedLMPreprocessor class]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm_preprocessor/#robertamaskedlmpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm_preprocessor/#from_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/models/roberta/roberta_masked_lm_preprocessor/#tokenizer-property" >}})
