---
title: Models API
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

### [Backbone]({{< relref "/docs/api/keras_nlp/base_classes/backbone/" >}})

- [Backbone class]({{< relref "/docs/api/keras_nlp/base_classes/backbone/#backbone-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/backbone/#from_preset-method" >}})
- [token\_embedding property]({{< relref "/docs/api/keras_nlp/base_classes/backbone/#token_embedding-property" >}})
- [enable\_lora method]({{< relref "/docs/api/keras_nlp/base_classes/backbone/#enable_lora-method" >}})
- [save\_lora\_weights method]({{< relref "/docs/api/keras_nlp/base_classes/backbone/#save_lora_weights-method" >}})
- [load\_lora\_weights method]({{< relref "/docs/api/keras_nlp/base_classes/backbone/#load_lora_weights-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/backbone/#save_to_preset-method" >}})

### [Task]({{< relref "/docs/api/keras_nlp/base_classes/task/" >}})

- [Task class]({{< relref "/docs/api/keras_nlp/base_classes/task/#task-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/task/#from_preset-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/task/#save_to_preset-method" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/base_classes/task/#preprocessor-property" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/base_classes/task/#backbone-property" >}})

### [Preprocessor]({{< relref "/docs/api/keras_nlp/base_classes/preprocessor/" >}})

- [Preprocessor class]({{< relref "/docs/api/keras_nlp/base_classes/preprocessor/#preprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/preprocessor/#from_preset-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/preprocessor/#save_to_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/base_classes/preprocessor/#tokenizer-property" >}})

### [CausalLM]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/" >}})

- [CausalLM class]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/#causallm-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/#from_preset-method" >}})
- [compile method]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/#compile-method" >}})
- [generate method]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/#generate-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/#save_to_preset-method" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/#preprocessor-property" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm/#backbone-property" >}})

### [CausalLMPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm_preprocessor/" >}})

- [CausalLMPreprocessor class]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm_preprocessor/#causallmpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm_preprocessor/#from_preset-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm_preprocessor/#save_to_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm_preprocessor/#tokenizer-property" >}})

### [Seq2SeqLM]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/" >}})

- [Seq2SeqLM class]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/#seq2seqlm-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/#from_preset-method" >}})
- [compile method]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/#compile-method" >}})
- [generate method]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/#generate-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/#save_to_preset-method" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/#preprocessor-property" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm/#backbone-property" >}})

### [Seq2SeqLMPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm_preprocessor/" >}})

- [Seq2SeqLMPreprocessor class]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm_preprocessor/#seq2seqlmpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm_preprocessor/#from_preset-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm_preprocessor/#save_to_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm_preprocessor/#tokenizer-property" >}})

### [TextClassifier]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier/" >}})

- [TextClassifier class]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier/#textclassifier-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier/#from_preset-method" >}})
- [compile method]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier/#compile-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier/#save_to_preset-method" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier/#preprocessor-property" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier/#backbone-property" >}})

### [TextClassifierPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier_preprocessor/" >}})

- [TextClassifierPreprocessor class]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier_preprocessor/#textclassifierpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier_preprocessor/#from_preset-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier_preprocessor/#save_to_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier_preprocessor/#tokenizer-property" >}})

### [MaskedLM]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm/" >}})

- [MaskedLM class]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm/#maskedlm-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm/#from_preset-method" >}})
- [compile method]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm/#compile-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm/#save_to_preset-method" >}})
- [preprocessor property]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm/#preprocessor-property" >}})
- [backbone property]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm/#backbone-property" >}})

### [MaskedLMPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm_preprocessor/" >}})

- [MaskedLMPreprocessor class]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm_preprocessor/#maskedlmpreprocessor-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm_preprocessor/#from_preset-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm_preprocessor/#save_to_preset-method" >}})
- [tokenizer property]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm_preprocessor/#tokenizer-property" >}})

### [upload\_preset]({{< relref "/docs/api/keras_nlp/base_classes/upload_preset/" >}})

- [upload\_preset function]({{< relref "/docs/api/keras_nlp/base_classes/upload_preset/#upload_preset-function" >}})
