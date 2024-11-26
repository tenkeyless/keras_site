---
title: KerasNLP Tokenizers
linkTitle: Tokenizers
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

Tokenizers convert raw string input into integer input suitable for a Keras `Embedding` layer.
They can also convert back from predicted integer sequences to raw string output.

All tokenizers subclass [`keras_nlp.tokenizers.Tokenizer`]({{< relref "/docs/api/keras_nlp/tokenizers/tokenizer#tokenizer-class" >}}), which in turn
subclasses [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}). Tokenizers should generally be applied inside a
[tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)
for training, and can be included inside a [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}) for inference.

### [Tokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/tokenizer/" >}})

- [Tokenizer class]({{< relref "/docs/api/keras_nlp/tokenizers/tokenizer/#tokenizer-class" >}})
- [from\_preset method]({{< relref "/docs/api/keras_nlp/tokenizers/tokenizer/#from_preset-method" >}})
- [save\_to\_preset method]({{< relref "/docs/api/keras_nlp/tokenizers/tokenizer/#save_to_preset-method" >}})

### [WordPieceTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/" >}})

- [WordPieceTokenizer class]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/#wordpiecetokenizer-class" >}})
- [tokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/#tokenize-method" >}})
- [detokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/#detokenize-method" >}})
- [get\_vocabulary method]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/#get_vocabulary-method" >}})
- [vocabulary\_size method]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/#vocabulary_size-method" >}})
- [token\_to\_id method]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/#token_to_id-method" >}})
- [id\_to\_token method]({{< relref "/docs/api/keras_nlp/tokenizers/word_piece_tokenizer/#id_to_token-method" >}})

### [SentencePieceTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/" >}})

- [SentencePieceTokenizer class]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/#sentencepiecetokenizer-class" >}})
- [tokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/#tokenize-method" >}})
- [detokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/#detokenize-method" >}})
- [get\_vocabulary method]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/#get_vocabulary-method" >}})
- [vocabulary\_size method]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/#vocabulary_size-method" >}})
- [token\_to\_id method]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/#token_to_id-method" >}})
- [id\_to\_token method]({{< relref "/docs/api/keras_nlp/tokenizers/sentence_piece_tokenizer/#id_to_token-method" >}})

### [BytePairTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/" >}})

- [BytePairTokenizer class]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/#bytepairtokenizer-class" >}})
- [tokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/#tokenize-method" >}})
- [detokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/#detokenize-method" >}})
- [get\_vocabulary method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/#get_vocabulary-method" >}})
- [vocabulary\_size method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/#vocabulary_size-method" >}})
- [token\_to\_id method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/#token_to_id-method" >}})
- [id\_to\_token method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_pair_tokenizer/#id_to_token-method" >}})

### [ByteTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/" >}})

- [ByteTokenizer class]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/#bytetokenizer-class" >}})
- [tokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/#tokenize-method" >}})
- [detokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/#detokenize-method" >}})
- [get\_vocabulary method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/#get_vocabulary-method" >}})
- [vocabulary\_size method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/#vocabulary_size-method" >}})
- [token\_to\_id method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/#token_to_id-method" >}})
- [id\_to\_token method]({{< relref "/docs/api/keras_nlp/tokenizers/byte_tokenizer/#id_to_token-method" >}})

### [UnicodeCodepointTokenizer]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/" >}})

- [UnicodeCodepointTokenizer class]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/#unicodecodepointtokenizer-class" >}})
- [tokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/#tokenize-method" >}})
- [detokenize method]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/#detokenize-method" >}})
- [get\_vocabulary method]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/#get_vocabulary-method" >}})
- [vocabulary\_size method]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/#vocabulary_size-method" >}})
- [token\_to\_id method]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/#token_to_id-method" >}})
- [id\_to\_token method]({{< relref "/docs/api/keras_nlp/tokenizers/unicode_codepoint_tokenizer/#id_to_token-method" >}})

### [compute\_word\_piece\_vocabulary function]({{< relref "/docs/api/keras_nlp/tokenizers/compute_word_piece_vocabulary/" >}})

- [compute\_word\_piece\_vocabulary function]({{< relref "/docs/api/keras_nlp/tokenizers/compute_word_piece_vocabulary/#compute_word_piece_vocabulary-function" >}})

### [compute\_sentence\_piece\_proto function]({{< relref "/docs/api/keras_nlp/tokenizers/compute_sentence_piece_proto/" >}})

- [compute\_sentence\_piece\_proto function]({{< relref "/docs/api/keras_nlp/tokenizers/compute_sentence_piece_proto/#compute_sentence_piece_proto-function" >}})
