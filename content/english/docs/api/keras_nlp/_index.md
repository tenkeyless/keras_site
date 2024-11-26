---
title: KerasNLP
toc: true
weight: 17
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

KerasNLP is a toolbox of modular building blocks ranging from pretrained
state-of-the-art models, to low-level Transformer Encoder layers. For an
introduction to the library see the [KerasNLP home page]({{< relref "/docs/keras_nlp" >}}). For a
high-level introduction to the API see our
[getting started guide]({{< relref "/docs/guides/keras_nlp/getting_started/" >}}).

### [Pretrained Models]({{< relref "/docs/api/keras_nlp/models/" >}})

- [Albert]({{< relref "/docs/api/keras_nlp/models/albert/" >}})
- [Bart]({{< relref "/docs/api/keras_nlp/models/bart/" >}})
- [Bert]({{< relref "/docs/api/keras_nlp/models/bert/" >}})
- [Bloom]({{< relref "/docs/api/keras_nlp/models/bloom/" >}})
- [DebertaV3]({{< relref "/docs/api/keras_nlp/models/deberta_v3/" >}})
- [DistilBert]({{< relref "/docs/api/keras_nlp/models/distil_bert/" >}})
- [Gemma]({{< relref "/docs/api/keras_nlp/models/gemma/" >}})
- [Electra]({{< relref "/docs/api/keras_nlp/models/electra/" >}})
- [Falcon]({{< relref "/docs/api/keras_nlp/models/falcon/" >}})
- [FNet]({{< relref "/docs/api/keras_nlp/models/f_net/" >}})
- [GPT2]({{< relref "/docs/api/keras_nlp/models/gpt2/" >}})
- [Llama]({{< relref "/docs/api/keras_nlp/models/llama/" >}})
- [Llama3]({{< relref "/docs/api/keras_nlp/models/llama3/" >}})
- [Mistral]({{< relref "/docs/api/keras_nlp/models/mistral/" >}})
- [OPT]({{< relref "/docs/api/keras_nlp/models/opt/" >}})
- [PaliGemma]({{< relref "/docs/api/keras_nlp/models/pali_gemma/" >}})
- [Phi3]({{< relref "/docs/api/keras_nlp/models/phi3/" >}})
- [Roberta]({{< relref "/docs/api/keras_nlp/models/roberta/" >}})
- [XLMRoberta]({{< relref "/docs/api/keras_nlp/models/xlm_roberta/" >}})

### [Models API]({{< relref "/docs/api/keras_nlp/base_classes/" >}})

- [Backbone]({{< relref "/docs/api/keras_nlp/base_classes/backbone" >}})
- [Task]({{< relref "/docs/api/keras_nlp/base_classes/task" >}})
- [Preprocessor]({{< relref "/docs/api/keras_nlp/base_classes/preprocessor" >}})
- [CausalLM]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm" >}})
- [CausalLMPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/causal_lm_preprocessor" >}})
- [Seq2SeqLM]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm" >}})
- [Seq2SeqLMPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/seq_2_seq_lm_preprocessor" >}})
- [TextClassifier]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier" >}})
- [TextClassifierPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/text_classifier_preprocessor" >}})
- [MaskedLM]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm" >}})
- [MaskedLMPreprocessor]({{< relref "/docs/api/keras_nlp/base_classes/masked_lm_preprocessor" >}})
- [upload\_preset]({{< relref "/docs/api/keras_nlp/base_classes/upload_preset" >}})

### [Tokenizers]({{< relref "/docs/api/keras_nlp/tokenizers/" >}})

- [Tokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/tokenizer" >}})
- [WordPieceTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer" >}})
- [SentencePieceTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer" >}})
- [BytePairTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer" >}})
- [ByteTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer" >}})
- [UnicodeCodepointTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer" >}})
- [compute\_word\_piece\_vocabulary function]({{< relref "/docs/api/keras_nlp/tokenizers/compute_word_piece_vocabulary" >}})
- [compute\_sentence\_piece\_proto function]({{< relref "/docs/api/keras_nlp/tokenizers/compute_sentence_piece_proto" >}})

### [Preprocessing Layers]({{< relref "/docs/api/keras_nlp/preprocessing_layers/" >}})

- [StartEndPacker layer]({{< relref "/docs/api/keras_nlp/preprocessing_layers/start_end_packer" >}})
- [MultiSegmentPacker layer]({{< relref "/docs/api/keras_nlp/preprocessing_layers/multi_segment_packer" >}})
- [RandomSwap layer]({{< relref "/docs/api/keras_nlp/preprocessing_layers/random_swap" >}})
- [RandomDeletion layer]({{< relref "/docs/api/keras_nlp/preprocessing_layers/random_deletion" >}})
- [MaskedLMMaskGenerator layer]({{< relref "/docs/api/keras_nlp/preprocessing_layers/masked_lm_mask_generator" >}})

### [Modeling Layers]({{< relref "/docs/api/keras_nlp/modeling_layers/" >}})

- [TransformerEncoder layer]({{< relref "/docs/api/keras_nlp/modeling_layers/transformer_encoder" >}})
- [TransformerDecoder layer]({{< relref "/docs/api/keras_nlp/modeling_layers/transformer_decoder" >}})
- [FNetEncoder layer]({{< relref "/docs/api/keras_nlp/modeling_layers/fnet_encoder" >}})
- [PositionEmbedding layer]({{< relref "/docs/api/keras_nlp/modeling_layers/position_embedding" >}})
- [RotaryEmbedding layer]({{< relref "/docs/api/keras_nlp/modeling_layers/rotary_embedding" >}})
- [SinePositionEncoding layer]({{< relref "/docs/api/keras_nlp/modeling_layers/sine_position_encoding" >}})
- [ReversibleEmbedding layer]({{< relref "/docs/api/keras_nlp/modeling_layers/reversible_embedding" >}})
- [TokenAndPositionEmbedding layer]({{< relref "/docs/api/keras_nlp/modeling_layers/token_and_position_embedding" >}})
- [AlibiBias layer]({{< relref "/docs/api/keras_nlp/modeling_layers/alibi_bias" >}})
- [MaskedLMHead layer]({{< relref "/docs/api/keras_nlp/modeling_layers/masked_lm_head" >}})
- [CachedMultiHeadAttention layer]({{< relref "/docs/api/keras_nlp/modeling_layers/cached_multi_head_attention" >}})

### [Samplers]({{< relref "/docs/api/keras_nlp/samplers/" >}})

- [Sampler base class]({{< relref "/docs/api/keras_nlp/samplers/samplers" >}})
- [BeamSampler]({{< relref "/docs/api/keras_nlp/samplers/beam_sampler" >}})
- [ContrastiveSampler]({{< relref "/docs/api/keras_nlp/samplers/contrastive_sampler" >}})
- [GreedySampler]({{< relref "/docs/api/keras_nlp/samplers/greedy_sampler" >}})
- [RandomSampler]({{< relref "/docs/api/keras_nlp/samplers/random_sampler" >}})
- [TopKSampler]({{< relref "/docs/api/keras_nlp/samplers/top_k_sampler" >}})
- [TopPSampler]({{< relref "/docs/api/keras_nlp/samplers/top_p_sampler" >}})

### [Metrics]({{< relref "/docs/api/keras_nlp/metrics/" >}})

- [Perplexity metric]({{< relref "/docs/api/keras_nlp/metrics/perplexity" >}})
