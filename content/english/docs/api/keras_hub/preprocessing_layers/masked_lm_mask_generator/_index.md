---
title: MaskedLMMaskGenerator layer
toc: true
weight: 7
type: docs
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/preprocessing/masked_lm_mask_generator.py#L16" >}}

### `MaskedLMMaskGenerator` class

`keras_hub.layers.MaskedLMMaskGenerator(     vocabulary_size,     mask_selection_rate,     mask_token_id,     mask_selection_length=None,     unselectable_token_ids=[0],     mask_token_rate=0.8,     random_token_rate=0.1,     **kwargs )`

Layer that applies language model masking.

This layer is useful for preparing inputs for masked language modeling (MaskedLM) tasks. It follows the masking strategy described in the [original BERT paper](https://arxiv.org/abs/1810.04805). Given tokenized text, it randomly selects certain number of tokens for masking. Then for each selected token, it has a chance (configurable) to be replaced by "mask token" or random token, or stay unchanged.

Input data should be passed as tensors, [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor)s, or lists. For batched input, inputs should be a list of lists or a rank two tensor. For unbatched inputs, each element should be a list or a rank one tensor.

This layer can be used with [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) to generate dynamic masks on the fly during training.

**Arguments**

- **vocabulary_size**: int, the size of the vocabulary.
- **mask_selection_rate**: float, the probability of a token is selected for masking.
- **mask_token_id**: int. The id of mask token.
- **mask_selection_length**: int. Maximum number of tokens selected for masking in each sequence. If set, the output `mask_positions`, `mask_ids` and `mask_weights` will be padded to dense tensors of length `mask_selection_length`, otherwise the output will be a RaggedTensor. Defaults to `None`.
- **unselectable_token_ids**: A list of tokens id that should not be considered eligible for masking. By default, we assume `0` corresponds to a padding token and ignore it. Defaults to `[0]`.
- **mask_token_rate**: float. `mask_token_rate` must be between 0 and 1 which indicates how often the mask_token is substituted for tokens selected for masking. Defaults to `0.8`.
- **random_token_rate**: float. `random_token_rate` must be between 0 and 1 which indicates how often a random token is substituted for tokens selected for masking. Note: mask_token_rate + random_token_rate <= 1, and for (1 - mask_token_rate - random_token_rate), the token will not be changed. Defaults to `0.1`.

**Returns**

- **A Dict with 4 keys**: token_ids: Tensor or RaggedTensor, has the same type and shape of input. Sequence after getting masked. mask_positions: Tensor, or RaggedTensor if `mask_selection_length` is None. The positions of token_ids getting masked. mask_ids: Tensor, or RaggedTensor if `mask_selection_length` is None. The original token ids at masked positions. mask_weights: Tensor, or RaggedTensor if `mask_selection_length` is None. `mask_weights` has the same shape as `mask_positions` and `mask_ids`. Each element in `mask_weights` should be 0 or 1, 1 means the corresponding position in `mask_positions` is an actual mask, 0 means it is a pad.

**Examples**

Basic usage.

`masker = keras_hub.layers.MaskedLMMaskGenerator(     vocabulary_size=10,     mask_selection_rate=0.2,     mask_token_id=0,     mask_selection_length=5 ) # Dense input. masker([1, 2, 3, 4, 5])  # Ragged input. masker([[1, 2], [1, 2, 3, 4]])`

Masking a batch that contains special tokens.

`pad_id, cls_id, sep_id, mask_id = 0, 1, 2, 3 batch = [     [cls_id,   4,    5,      6, sep_id,    7,    8, sep_id, pad_id, pad_id],     [cls_id,   4,    5, sep_id,      6,    7,    8,      9, sep_id, pad_id], ]  masker = keras_hub.layers.MaskedLMMaskGenerator(     vocabulary_size = 10,     mask_selection_rate = 0.2,     mask_selection_length = 5,     mask_token_id = mask_id,     unselectable_token_ids = [         cls_id,         sep_id,         pad_id,     ] ) masker(batch)`

---
