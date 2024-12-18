---
title: Natural Language Processing
toc: true
weight: 2
---

{{< keras/original checkedAt="2024-11-22" >}}

### Text classification

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------ | ------------- |
| ★       | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/text_classification_from_scratch" >}}            | 2019/11/06   | 2020/05/17    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/active_learning_review_classification" >}}       | 2021/10/29   | 2024/05/08    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/fnet_classification_with_keras_hub" >}}          | 2022/06/01   | 2022/12/21    |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/multi_label_classification" >}}                  | 2020/09/25   | 2020/12/23    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/text_classification_with_transformer" >}}        | 2020/05/10   | 2024/01/18    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/text_classification_with_switch_transformer" >}} | 2020/05/10   | 2021/02/15    |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/tweet-classification-using-tfdf" >}}             | 2022/09/05   | 2022/09/05    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/pretrained_word_embeddings" >}}                  | 2020/05/05   | 2020/05/05    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/bidirectional_lstm_imdb" >}}                     | 2020/05/03   | 2020/05/03    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/data_parallel_training_with_keras_hub" >}}       | 2023/07/07   | 2023/07/07    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/active_learning_review_classification" >}}       | 2021/10/29   | 2024/05/08      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/text_classification_with_transformer" >}}        | 2020/05/10   | 2024/01/18      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/data_parallel_training_with_keras_hub" >}}       | 2023/07/07   | 2023/07/07      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/fnet_classification_with_keras_hub" >}}          | 2022/06/01   | 2022/12/21      |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/tweet-classification-using-tfdf" >}}             | 2022/09/05   | 2022/09/05      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/text_classification_with_switch_transformer" >}} | 2020/05/10   | 2021/02/15      |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/multi_label_classification" >}}                  | 2020/09/25   | 2020/12/23      |
| ★       | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/text_classification_from_scratch" >}}            | 2019/11/06   | 2020/05/17      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/pretrained_word_embeddings" >}}                  | 2020/05/05   | 2020/05/05      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/bidirectional_lstm_imdb" >}}                     | 2020/05/03   | 2020/05/03      |

### Machine translation

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------ | ------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/neural_machine_translation_with_keras_hub" >}}   | 2022/05/26   | 2024/04/30    |
| ★       | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/neural_machine_translation_with_transformer" >}} | 2021/05/26   | 2023/02/25    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/lstm_seq2seq" >}}                                | 2017/09/29   | 2023/11/22    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/neural_machine_translation_with_keras_hub" >}}   | 2022/05/26   | 2024/04/30      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/lstm_seq2seq" >}}                                | 2017/09/29   | 2023/11/22      |
| ★       | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/neural_machine_translation_with_transformer" >}} | 2021/05/26   | 2023/02/25      |

### Entailment prediction

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------ | ------------- |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/multimodal_entailment" >}} | 2021/08/08   | 2021/08/15    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/multimodal_entailment" >}} | 2021/08/08   | 2021/08/15      |

### Named entity recognition

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------ | ------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/ner_transformers" >}} | 2021/06/23   | 2024/04/05    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/ner_transformers" >}} | 2021/06/23   | 2024/04/05      |

### Sequence-to-sequence

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------ | ------------- |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/text_extraction_with_bert" >}} | 2020/05/23   | 2020/05/23    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/addition_rnn" >}}              | 2015/08/17   | 2024/02/13    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/addition_rnn" >}}              | 2015/08/17   | 2024/02/13      |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/text_extraction_with_bert" >}} | 2020/05/23   | 2020/05/23      |

### Text similarity search

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------ | ------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/semantic_similarity_with_keras_hub" >}} | 2023/02/25   | 2023/02/25    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/semantic_similarity_with_bert" >}}      | 2020/08/15   | 2020/08/29    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/sentence_embeddings_with_sbert" >}}     | 2023/07/14   | 2023/07/14    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/sentence_embeddings_with_sbert" >}}     | 2023/07/14   | 2023/07/14      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/semantic_similarity_with_keras_hub" >}} | 2023/02/25   | 2023/02/25      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/semantic_similarity_with_bert" >}}      | 2020/08/15   | 2020/08/29      |

### Language modeling

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------ | ------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/masked_language_modeling" >}}            | 2020/09/18   | 2024/03/15    |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/abstractive_summarization_with_bart" >}} | 2023/07/08   | 2024/03/20    |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/pretraining_BERT" >}}                    | 2022/07/01   | 2022/08/27    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/masked_language_modeling" >}}            | 2020/09/18   | 2024/03/15      |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}}               | {{< titledRelref "/docs/examples/nlp/abstractive_summarization_with_bart" >}} | 2023/07/08   | 2024/03/20      |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/pretraining_BERT" >}}                    | 2022/07/01   | 2022/08/27      |

### Parameter efficient fine-tuning

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------ | ------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora" >}} | 2023/05/27   | 2023/05/27    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------ | --------------- |
|         | {{< hextra/hero-button text="V3" style="background: rgb(23, 132, 133);pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora" >}} | 2023/05/27   | 2023/05/27      |

### Other

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------ | ------------- |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/mlm_training_tpus" >}}                           | 2023/05/21   | 2023/05/21    |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/multiple_choice_task_with_transfer_learning" >}} | 2023/09/14   | 2023/09/14    |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/question_answering" >}}                          | 2022/01/13   | 2022/01/13    |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/t5_hf_summarization" >}}                         | 2022/07/04   | 2022/08/28    |

| {{< t f_example_starter >}} | {{< t f_example_version >}}                                                                                                        | {{< t f_example_title >}}                                                                                 | {{< t f_example_date_created >}} | {{< t f_example_last_modified >}} ▼ |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------ | --------------- |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/multiple_choice_task_with_transfer_learning" >}} | 2023/09/14   | 2023/09/14      |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/mlm_training_tpus" >}}                           | 2023/05/21   | 2023/05/21      |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/t5_hf_summarization" >}}                         | 2022/07/04   | 2022/08/28      |
|         | {{< hextra/hero-button text="V2" style="background: rgb(255 237 183); color: black; pointer-events: none; padding: 0.1em 1em;" >}} | {{< titledRelref "/docs/examples/nlp/question_answering" >}}                          | 2022/01/13   | 2022/01/13      |
