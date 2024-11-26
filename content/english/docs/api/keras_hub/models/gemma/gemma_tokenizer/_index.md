---
title: GemmaTokenizer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/gemma/gemma_tokenizer.py#L8" >}}

### `GemmaTokenizer` class

```python
keras_hub.tokenizers.GemmaTokenizer(proto, **kwargs)
```

Gemma tokenizer layer based on SentencePiece.

This tokenizer class will tokenize raw strings into integer sequences and
is based on [`keras_hub.tokenizers.SentencePieceTokenizer`]({{< relref "/docs/api/keras_hub/tokenizers/sentence_piece_tokenizer#sentencepiecetokenizer-class" >}}). Unlike the
underlying tokenizer, it will check for all special tokens needed by
Gemma models and provides a `from_preset()` method to automatically
download a matching vocabulary for a Gemma preset.

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
tokenizer = keras_hub.models.GemmaTokenizer.from_preset("gemma_2b_en")
tokenizer("The quick brown fox jumped.")
# Batched input.
tokenizer(["The quick brown fox jumped.", "The fox slept."])
# Detokenization.
tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
# Custom vocabulary.
bytes_io = io.BytesIO()
ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
sentencepiece.SentencePieceTrainer.train(
    sentence_iterator=ds.as_numpy_iterator(),
    model_writer=bytes_io,
    vocab_size=8,
    model_type="WORD",
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
    pad_piece="<pad>",
    bos_piece="<bos>",
    eos_piece="<eos>",
    unk_piece="<unk>",
)
tokenizer = keras_hub.models.GemmaTokenizer(
    proto=bytes_io.getvalue(),
)
tokenizer("The quick brown fox jumped.")
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/tokenizers/tokenizer.py#L213" >}}

### `from_preset` method

```python
GemmaTokenizer.from_preset(preset, config_file="tokenizer.json", **kwargs)
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

| Preset name                   | Parameters | Description                                                                                                                                                                |
| ----------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| gemma_2b_en                   | 2.51B      | 2 billion parameter, 18-layer, base Gemma model.                                                                                                                           |
| gemma_instruct_2b_en          | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model.                                                                                                              |
| gemma_1.1_instruct_2b_en      | 2.51B      | 2 billion parameter, 18-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                       |
| code_gemma_1.1_2b_en          | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion. The 1.1 update improves model quality. |
| code_gemma_2b_en              | 2.51B      | 2 billion parameter, 18-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                        |
| gemma_7b_en                   | 8.54B      | 7 billion parameter, 28-layer, base Gemma model.                                                                                                                           |
| gemma_instruct_7b_en          | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model.                                                                                                              |
| gemma_1.1_instruct_7b_en      | 8.54B      | 7 billion parameter, 28-layer, instruction tuned Gemma model. The 1.1 update improves model quality.                                                                       |
| code_gemma_7b_en              | 8.54B      | 7 billion parameter, 28-layer, CodeGemma model. This model has been trained on a fill-in-the-middle (FIM) task for code completion.                                        |
| code_gemma_instruct_7b_en     | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code.                                          |
| code_gemma_1.1_instruct_7b_en | 8.54B      | 7 billion parameter, 28-layer, instruction tuned CodeGemma model. This model has been trained for chat use cases related to code. The 1.1 update improves model quality.   |
| gemma2_2b_en                  | 2.61B      | 2 billion parameter, 26-layer, base Gemma model.                                                                                                                           |
| gemma2_instruct_2b_en         | 2.61B      | 2 billion parameter, 26-layer, instruction tuned Gemma model.                                                                                                              |
| gemma2_9b_en                  | 9.24B      | 9 billion parameter, 42-layer, base Gemma model.                                                                                                                           |
| gemma2_instruct_9b_en         | 9.24B      | 9 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                              |
| gemma2_27b_en                 | 27.23B     | 27 billion parameter, 42-layer, base Gemma model.                                                                                                                          |
| gemma2_instruct_27b_en        | 27.23B     | 27 billion parameter, 42-layer, instruction tuned Gemma model.                                                                                                             |
| shieldgemma_2b_en             | 2.61B      | 2 billion parameter, 26-layer, ShieldGemma model.                                                                                                                          |
| shieldgemma_9b_en             | 9.24B      | 9 billion parameter, 42-layer, ShieldGemma model.                                                                                                                          |
| shieldgemma_27b_en            | 27.23B     | 27 billion parameter, 42-layer, ShieldGemma model.                                                                                                                         |
| pali_gemma_3b_mix_224         | 2.92B      | image size 224, mix fine tuned, text sequence length is 256                                                                                                                |
| pali_gemma_3b_mix_448         | 2.92B      | image size 448, mix fine tuned, text sequence length is 512                                                                                                                |
| pali_gemma_3b_224             | 2.92B      | image size 224, pre trained, text sequence length is 128                                                                                                                   |
| pali_gemma_3b_448             | 2.92B      | image size 448, pre trained, text sequence length is 512                                                                                                                   |
| pali_gemma_3b_896             | 2.93B      | image size 896, pre trained, text sequence length is 512                                                                                                                   |
