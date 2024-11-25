---
title: StartEndPacker layer
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/preprocessing/start_end_packer.py#L14" >}}

### `StartEndPacker` class

`keras_nlp.layers.StartEndPacker(     sequence_length,     start_value=None,     end_value=None,     pad_value=None,     return_padding_mask=False,     name=None,     **kwargs )`

Adds start and end tokens to a sequence and pads to a fixed length.

This layer is useful when tokenizing inputs for tasks like translation, where each sequence should include a start and end marker. It should be called after tokenization. The layer will first trim inputs to fit, then add start/end tokens, and finally pad, if necessary, to `sequence_length`.

Input data should be passed as tensors, [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor)s, or lists. For batched input, inputs should be a list of lists or a rank two tensor. For unbatched inputs, each element should be a list or a rank one tensor.

**Arguments**

- **sequence_length**: int. The desired output length.
- **start_value**: int/str/list/tuple. The ID(s) or token(s) that are to be placed at the start of each sequence. The dtype must match the dtype of the input tensors to the layer. If `None`, no start value will be added.
- **end_value**: int/str/list/tuple. The ID(s) or token(s) that are to be placed at the end of each input segment. The dtype must match the dtype of the input tensors to the layer. If `None`, no end value will be added.
- **pad_value**: int/str. The ID or token that is to be placed into the unused positions after the last segment in the sequence. If `None`, 0 or "" will be added depending on the dtype of the input tensor.
- **return_padding_mask**: bool. Whether to return a boolean padding mask of all locations that are filled in with the `pad_value`.

**Call arguments**

- **inputs**: A [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor), [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor), or list of python strings.
- **sequence_length**: Pass to override the configured `sequence_length` of the layer.
- **add_start_value**: Pass `False` to not append a start value for this input.
- **add_end_value**: Pass `False` to not append an end value for this input.

**Examples**

Unbatched input (int).

`>>> inputs = [5, 6, 7] >>> start_end_packer = keras_hub.layers.StartEndPacker( ...     sequence_length=7, start_value=1, end_value=2, ... ) >>> outputs = start_end_packer(inputs) >>> np.array(outputs) array([1, 5, 6, 7, 2, 0, 0], dtype=int32)`

Batched input (int).

`>>> inputs = [[5, 6, 7], [8, 9, 10, 11, 12, 13, 14]] >>> start_end_packer = keras_hub.layers.StartEndPacker( ...     sequence_length=6, start_value=1, end_value=2, ... ) >>> outputs = start_end_packer(inputs) >>> np.array(outputs) array([[ 1,  5,  6,  7,  2,  0],        [ 1,  8,  9, 10, 11,  2]], dtype=int32)`

Unbatched input (str).

`>>> inputs = tf.constant(["this", "is", "fun"]) >>> start_end_packer = keras_hub.layers.StartEndPacker( ...     sequence_length=6, start_value="<s>", end_value="</s>", ...     pad_value="<pad>" ... ) >>> outputs = start_end_packer(inputs) >>> np.array(outputs).astype("U") array(['<s>', 'this', 'is', 'fun', '</s>', '<pad>'], dtype='<U5')`

Batched input (str).

`>>> inputs = tf.ragged.constant([["this", "is", "fun"], ["awesome"]]) >>> start_end_packer = keras_hub.layers.StartEndPacker( ...     sequence_length=6, start_value="<s>", end_value="</s>", ...     pad_value="<pad>" ... ) >>> outputs = start_end_packer(inputs) >>> np.array(outputs).astype("U") array([['<s>', 'this', 'is', 'fun', '</s>', '<pad>'],        ['<s>', 'awesome', '</s>', '<pad>', '<pad>', '<pad>']], dtype='<U7')`

Multiple start tokens.

`>>> inputs = tf.ragged.constant([["this", "is", "fun"], ["awesome"]]) >>> start_end_packer = keras_hub.layers.StartEndPacker( ...     sequence_length=6, start_value=["</s>", "<s>"], end_value="</s>", ...     pad_value="<pad>" ... ) >>> outputs = start_end_packer(inputs) >>> np.array(outputs).astype("U") array([['</s>', '<s>', 'this', 'is', 'fun', '</s>'],        ['</s>', '<s>', 'awesome', '</s>', '<pad>', '<pad>']], dtype='<U7')`

---
