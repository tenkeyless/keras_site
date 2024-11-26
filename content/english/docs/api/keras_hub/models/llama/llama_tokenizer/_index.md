---
title: LlamaTokenizer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/llama/llama_tokenizer.py#L8" >}}

### `LlamaTokenizer` class

```python
keras_hub.tokenizers.LlamaTokenizer(proto, **kwargs)
```

Llama tokenizer layer based on SentencePiece.

This tokenizer class will tokenize raw strings into integer sequences and
is based on [`keras_hub.tokenizers.SentencePieceTokenizer`]({{< relref "/docs/api/keras_hub/tokenizers/sentence_piece_tokenizer#sentencepiecetokenizer-class" >}}). Unlike the
underlying tokenizer, it will check for all special tokens needed by
Llama models and provides a `from_preset()` method to automatically
download a matching vocabulary for a Llama preset.

If input is a batch of strings (rank > 0), the layer will output a
[`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) where the last dimension of the output is ragged.

If input is a scalar string (rank == 0), the layer will output a dense
[`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) with static shape `[None]`.

**Arguments**

- **proto**: Either a `string` path to a SentencePiece proto file, or a
  `bytes` object with a serialized SentencePiece proto. See the
  [SentencePiece repository](https://github.com/google/sentencepiece)
  for more details on the format.

**Examples**

```python
# Unbatched input.
tokenizer = keras_hub.models.LlamaTokenizer.from_preset(
    "llama_7b_en",
)
tokenizer("The quick brown fox jumped.")
# Batched input.
tokenizer(["The quick brown fox jumped.", "The fox slept."])
# Detokenization.
tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L213" >}}

### `from_preset` method

```python
LlamaTokenizer.from_preset(preset, config_file="tokenizer.json", **kwargs)
```

Instantiate a `keras_hub.models.Tokenizer` from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Tokenizer` subclass, you can run `cls.presets.keys()` to list
all built-in presets available on the class.

This constructor can be called in one of two ways. Either from the base
class like `keras_hub.models.Tokenizer.from_preset()`, or from
a model class like `keras_hub.models.GemmaTokenizer.from_preset()`.
If calling from the base class, the subclass of the returning object
will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the
  model architecture. If `False`, the weights will be randomly
  initialized.

**Examples**

```python
# Load a preset tokenizer.
tokenizer = keras_hub.tokenizer.Tokenizer.from_preset("bert_base_en")
# Tokenize some input.
tokenizer("The quick brown fox tripped.")
# Detokenize some input.
tokenizer.detokenize([5, 6, 7, 8, 9])
```

| Preset name                | Parameters | Description                                                                                                   |
| -------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| llama2_7b_en               | 6.74B      | 7 billion parameter, 32-layer, base LLaMA 2 model.                                                            |
| llama2_7b_en_int8          | 6.74B      | 7 billion parameter, 32-layer, base LLaMA 2 model with activation and weights quantized to int8.              |
| llama2_instruct_7b_en      | 6.74B      | 7 billion parameter, 32-layer, instruction tuned LLaMA 2 model.                                               |
| llama2_instruct_7b_en_int8 | 6.74B      | 7 billion parameter, 32-layer, instruction tuned LLaMA 2 model with activation and weights quantized to int8. |
| vicuna_1.5_7b_en           | 6.74B      | 7 billion parameter, 32-layer, instruction tuned Vicuna v1.5 model.                                           |
| llama3_8b_en               | 8.03B      | 8 billion parameter, 32-layer, base LLaMA 3 model.                                                            |
| llama3_8b_en_int8          | 8.03B      | 8 billion parameter, 32-layer, base LLaMA 3 model with activation and weights quantized to int8.              |
| llama3_instruct_8b_en      | 8.03B      | 8 billion parameter, 32-layer, instruction tuned LLaMA 3 model.                                               |
| llama3_instruct_8b_en_int8 | 8.03B      | 8 billion parameter, 32-layer, instruction tuned LLaMA 3 model with activation and weights quantized to int8. |
