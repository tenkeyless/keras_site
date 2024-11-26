---
title: Bert
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

Models, tokenizers, and preprocessing layers for BERT,
as described in ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805).

For a full list of available **presets**, see the
[models page]({{< relref "/docs/api/keras_nlp/models" >}}).

### [BertTokenizer]({{< relref "/docs/api/keras_nlp/models/bert/bert_tokenizer/" >}})

- [BertTokenizer class]({{< relref "/docs/api/keras_nlp/models/bert/bert_tokenizer/#berttokenizer-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/bert/bert_tokenizer/#from_preset-method" >}})

### [BertBackbone model]({{< relref "/docs/api/keras_nlp/models/bert/bert_backbone/" >}})

- [BertBackbone class]({{< relref "/docs/api/keras_nlp/models/bert/bert_backbone/#bertbackbone-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/bert/bert_backbone/#from_preset-method" >}})
- [token\_embedding property]({{< relref "/docs/api/keras_nlp/models/bert/bert_backbone/#token_embedding-property" >}})

### [BertTextClassifier model]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier/" >}})

- [BertTextClassifier class]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier/#berttextclassifier-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier/#from_preset-method" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier/#backbone-property" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier/#preprocessor-property" >}})

### [BertTextClassifierPreprocessor layer]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier_preprocessor/" >}})

- [BertTextClassifierPreprocessor class]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier_preprocessor/#berttextclassifierpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier_preprocessor/#from_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/models/bert/bert_text_classifier_preprocessor/#tokenizer-property" >}})

### [BertMaskedLM model]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm/" >}})

- [BertMaskedLM class]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm/#bertmaskedlm-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm/#from_preset-method" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm/#backbone-property" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm/#preprocessor-property" >}})

### [BertMaskedLMPreprocessor layer]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm_preprocessor/" >}})

- [BertMaskedLMPreprocessor class]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm_preprocessor/#bertmaskedlmpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm_preprocessor/#from_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/models/bert/bert_masked_lm_preprocessor/#tokenizer-property" >}})
