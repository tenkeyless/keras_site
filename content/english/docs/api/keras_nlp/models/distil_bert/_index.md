---
title: DistilBERT
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

Models, tokenizers, and preprocessing layers for DistilBERT,
as described in ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108).

For a full list of available **presets**, see the
[models page]({{< relref "/docs/api/keras_nlp/models" >}}).

### [DistilBertTokenizer]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_tokenizer/" >}})

- [DistilBertTokenizer class]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_tokenizer/#distilberttokenizer-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_tokenizer/#from_preset-method" >}})

### [DistilBertBackbone model]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_backbone/" >}})

- [DistilBertBackbone class]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_backbone/#distilbertbackbone-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_backbone/#from_preset-method" >}})
- [token\_embedding property]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_backbone/#token_embedding-property" >}})

### [DistilBertTextClassifier model]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier/" >}})

- [DistilBertTextClassifier class]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier/#distilberttextclassifier-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier/#from_preset-method" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier/#backbone-property" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier/#preprocessor-property" >}})

### [DistilBertTextClassifierPreprocessor layer]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier_preprocessor/" >}})

- [DistilBertTextClassifierPreprocessor class]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier_preprocessor/#distilberttextclassifierpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier_preprocessor/#from_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_text_classifier_preprocessor/#tokenizer-property" >}})

### [DistilBertMaskedLM model]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm/" >}})

- [DistilBertMaskedLM class]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm/#distilbertmaskedlm-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm/#from_preset-method" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm/#backbone-property" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm/#preprocessor-property" >}})

### [DistilBertMaskedLMPreprocessor layer]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm_preprocessor/" >}})

- [DistilBertMaskedLMPreprocessor class]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm_preprocessor/#distilbertmaskedlmpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm_preprocessor/#from_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/models/distil_bert/distil_bert_masked_lm_preprocessor/#tokenizer-property" >}})
