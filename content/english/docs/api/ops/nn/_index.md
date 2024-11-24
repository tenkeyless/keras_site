---
title: NN ops
toc: true
weight: 2
type: docs
---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L754)

### `average_pool` function

`keras.ops.average_pool(     inputs, pool_size, strides=None, padding="valid", data_format=None )`

Average pooling operation.

**Arguments**

- **inputs**: Tensor of rank N+2. `inputs` has shape `(batch_size,) + inputs_spatial_shape + (num_channels,)` if `data_format="channels_last"`, or `(batch_size, num_channels) + inputs_spatial_shape` if `data_format="channels_first"`. Pooling happens over the spatial dimensions only.
- **pool_size**: int or tuple/list of integers of size `len(inputs_spatial_shape)`, specifying the size of the pooling window for each spatial dimension of the input tensor. If `pool_size` is int, then every spatial dimension shares the same `pool_size`.
- **strides**: int or tuple/list of integers of size `len(inputs_spatial_shape)`. The stride of the sliding window for each spatial dimension of the input tensor. If `strides` is int, then every spatial dimension shares the same `strides`.
- **padding**: string, either `"valid"` or `"same"`. `"valid"` means no padding is applied, and `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input when `strides=1`.
- **data_format**: A string, either `"channels_last"` or `"channels_first"`. `data_format` determines the ordering of the dimensions in the inputs. If `data_format="channels_last"`, `inputs` is of shape `(batch_size, ..., channels)` while if `data_format="channels_first"`, `inputs` is of shape `(batch_size, channels, ...)`.

**Returns**

A tensor of rank N+2, the result of the average pooling operation.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1746)

### `batch_normalization` function

`keras.ops.batch_normalization(     x, mean, variance, axis, offset=None, scale=None, epsilon=0.001 )`

Normalizes `x` by `mean` and `variance`.

This op is typically used by the batch normalization step in a neural network. It normalizes the input tensor along the given axis.

**Arguments**

- **x**: Input tensor.
- **mean**: A mean vector of the same length as the `axis` dimension of the input thensor.
- **variance**: A variance vector of the same length as the `axis` dimension of the input tensor.
- **axis**: Integer, the axis that should be normalized.
- **offset**: An offset vector of the same length as the `axis` dimension of the input tensor. If not `None`, `offset` is added to the normalized tensor. Defaults to `None`.
- **scale**: A scale vector of the same length as the `axis` dimension of the input tensor. If not `None`, the normalized tensor is multiplied by `scale`. Defaults to `None`.
- **epsilon**: Small float added to variance to avoid dividing by zero. Defaults to 1e-3.

**Returns**

The normalized tensor.

**Example**

`>>> x = keras.ops.convert_to_tensor( ...     [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]] ... ) >>> keras.ops.batch_normalization( ...     x, ...     mean=[0.4, 0.5, 0.6], ...     variance=[0.67, 0.67, 0.67], ...     axis=-1 ... ) array([[-3.6624e-01, -3.6624e-01, -3.6624e-01],        [-4.6445e-09,  0.0000e+00, -1.8578e-08],        [ 3.6624e-01,  3.6624e-01,  3.6624e-01]])`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1348)

### `binary_crossentropy` function

`keras.ops.binary_crossentropy(target, output, from_logits=False)`

Computes binary cross-entropy loss between target and output tensor.

The binary cross-entropy loss is commonly used in binary classification tasks where each input sample belongs to one of the two classes. It measures the dissimilarity between the target and output probabilities or logits.

**Arguments**

- **target**: The target tensor representing the true binary labels. Its shape should match the shape of the `output` tensor.
- **output**: The output tensor representing the predicted probabilities or logits. Its shape should match the shape of the `target` tensor.
- **from_logits**: (optional) Whether `output` is a tensor of logits or probabilities. Set it to `True` if `output` represents logits; otherwise, set it to `False` if `output` represents probabilities. Defaults to `False`.

**Returns**

- **Integer tensor**: The computed binary cross-entropy loss between `target` and `output`.

**Example**

`>>> target = keras.ops.convert_to_tensor([0, 1, 1, 0]) >>> output = keras.ops.convert_to_tensor([0.1, 0.9, 0.8, 0.2]) >>> binary_crossentropy(target, output) array([0.10536054 0.10536054 0.22314355 0.22314355],       shape=(4,), dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1422)

### `categorical_crossentropy` function

`keras.ops.categorical_crossentropy(target, output, from_logits=False, axis=-1)`

Computes categorical cross-entropy loss between target and output tensor.

The categorical cross-entropy loss is commonly used in multi-class classification tasks where each input sample can belong to one of multiple classes. It measures the dissimilarity between the target and output probabilities or logits.

**Arguments**

- **target**: The target tensor representing the true categorical labels. Its shape should match the shape of the `output` tensor except for the last dimension.
- **output**: The output tensor representing the predicted probabilities or logits. Its shape should match the shape of the `target` tensor except for the last dimension.
- **from_logits**: (optional) Whether `output` is a tensor of logits or probabilities. Set it to `True` if `output` represents logits; otherwise, set it to `False` if `output` represents probabilities. Defaults to `False`.
- **axis**: (optional) The axis along which the categorical cross-entropy is computed. Defaults to `-1`, which corresponds to the last dimension of the tensors.

**Returns**

- **Integer tensor**: The computed categorical cross-entropy loss between `target` and `output`.

**Example**

`>>> target = keras.ops.convert_to_tensor( ... [[1, 0, 0], ...  [0, 1, 0], ...  [0, 0, 1]]) >>> output = keras.ops.convert_to_tensor( ... [[0.9, 0.05, 0.05], ...  [0.1, 0.8, 0.1], ...  [0.2, 0.3, 0.5]]) >>> categorical_crossentropy(target, output) array([0.10536054 0.22314355 0.6931472 ], shape=(3,), dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L850)

### `conv` function

`keras.ops.conv(     inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1 )`

General N-D convolution.

This ops supports 1D, 2D and 3D convolution.

**Arguments**

- **inputs**: Tensor of rank N+2. `inputs` has shape `(batch_size,) + inputs_spatial_shape + (num_channels,)` if `data_format="channels_last"`, or `(batch_size, num_channels) + inputs_spatial_shape` if `data_format="channels_first"`.
- **kernel**: Tensor of rank N+2. `kernel` has shape `(kernel_spatial_shape, num_input_channels, num_output_channels)`. `num_input_channels` should match the number of channels in `inputs`.
- **strides**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the strides of the convolution along each spatial dimension. If `strides` is int, then every spatial dimension shares the same `strides`.
- **padding**: string, either `"valid"` or `"same"`. `"valid"` means no padding is applied, and `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input when `strides=1`.
- **data_format**: A string, either `"channels_last"` or `"channels_first"`. `data_format` determines the ordering of the dimensions in the inputs. If `data_format="channels_last"`, `inputs` is of shape `(batch_size, ..., channels)` while if `data_format="channels_first"`, `inputs` is of shape `(batch_size, channels, ...)`.
- **dilation_rate**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the dilation rate to use for dilated convolution. If `dilation_rate` is int, then every spatial dimension shares the same `dilation_rate`.

**Returns**

A tensor of rank N+2, the result of the conv operation.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1174)

### `conv_transpose` function

`keras.ops.conv_transpose(     inputs,     kernel,     strides,     padding="valid",     output_padding=None,     data_format=None,     dilation_rate=1, )`

General N-D convolution transpose.

Also known as de-convolution. This ops supports 1D, 2D and 3D convolution.

**Arguments**

- **inputs**: Tensor of rank N+2. `inputs` has shape `(batch_size,) + inputs_spatial_shape + (num_channels,)` if `data_format="channels_last"`, or `(batch_size, num_channels) + inputs_spatial_shape` if `data_format="channels_first"`.
- **kernel**: Tensor of rank N+2. `kernel` has shape \[kernel_spatial_shape, num_output_channels, num_input_channels\], `num_input_channels` should match the number of channels in `inputs`.
- **strides**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the strides of the convolution along each spatial dimension. If `strides` is int, then every spatial dimension shares the same `strides`.
- **padding**: string, either `"valid"` or `"same"`. `"valid"` means no padding is applied, and `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input when `strides=1`.
- **output_padding**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to `None` (default), the output shape is inferred.
- **data_format**: A string, either `"channels_last"` or `"channels_first"`. `data_format` determines the ordering of the dimensions in the inputs. If `data_format="channels_last"`, `inputs` is of shape `(batch_size, ..., channels)` while if `data_format="channels_first"`, `inputs` is of shape `(batch_size, channels, ...)`.
- **dilation_rate**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the dilation rate to use for dilated convolution. If `dilation_rate` is int, then every spatial dimension shares the same `dilation_rate`.

**Returns**

A tensor of rank N+2, the result of the conv operation.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1910)

### `ctc_decode` function

`keras.ops.ctc_decode(     inputs,     sequence_lengths,     strategy="greedy",     beam_width=100,     top_paths=1,     merge_repeated=True,     mask_index=0, )`

Decodes the output of a CTC model.

**Arguments**

- **inputs**: A tensor of shape `(batch_size, max_length, num_classes)` containing the logits (the output of the model). They should _not_ be normalized via softmax.
- **sequence_lengths**: A tensor of shape `(batch_size,)` containing the sequence lengths for the batch.
- **strategy**: A string for the decoding strategy. Supported values are `"greedy"` and `"beam_search"`.
- **beam_width**: An integer scalar beam width used in beam search. Defaults to 100.
- **top_paths**: An integer scalar, the number of top paths to return. Defaults to 1.
- **merge_repeated**: A boolean scalar, whether to merge repeated labels in the output. Defaults to `True`.
- **mask_index**: An integer scalar, the index of the mask character in the vocabulary. Defaults to `0`.

**Returns**

- **A tuple containing**:
- The tensor representing the list of decoded sequences. If `strategy="greedy"`, the shape is `(1, batch_size, max_length)`. If `strategy="beam_search"`, the shape is `(top_paths, batch_size, max_length)`. Note that: `-1` indicates the blank label.
- If `strategy="greedy"`, a tensor of shape `(batch_size, 1)` representing the negative of the sum of the probability logits for each sequence. If `strategy="beam_seatch"`, a tensor of shape `(batch_size, top_paths)` representing the log probability for each sequence.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1837)

### `ctc_loss` function

`keras.ops.ctc_loss(target, output, target_length, output_length, mask_index=0)`

CTC (Connectionist Temporal Classification) loss.

**Arguments**

- **target**: A tensor of shape `(batch_size, max_length)` containing the true labels in integer format.
- **output**: A tensor of shape `(batch_size, max_length, num_classes)` containing logits (the output of your model).
- **target_length**: A tensor of shape `(batch_size,)` containing the true label lengths.
- **output_length**: A tensor of shape `(batch_size,)` containing the output lengths.
- **mask_index**: The index of the mask character in the vocabulary. Defaults to `0`.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L943)

### `depthwise_conv` function

`keras.ops.depthwise_conv(     inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1 )`

General N-D depthwise convolution.

This ops supports 1D and 2D depthwise convolution.

**Arguments**

- **inputs**: Tensor of rank N+2. `inputs` has shape `(batch_size,) + inputs_spatial_shape + (num_channels,)` if `data_format="channels_last"`, or `(batch_size, num_channels) + inputs_spatial_shape` if `data_format="channels_first"`.
- **kernel**: Tensor of rank N+2. `kernel` has shape \[kernel_spatial_shape, num_input_channels, num_channels_multiplier\], `num_input_channels` should match the number of channels in `inputs`.
- **strides**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the strides of the convolution along each spatial dimension. If `strides` is int, then every spatial dimension shares the same `strides`.
- **padding**: string, either `"valid"` or `"same"`. `"valid"` means no padding is applied, and `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input when `strides=1`.
- **data_format**: A string, either `"channels_last"` or `"channels_first"`. `data_format` determines the ordering of the dimensions in the inputs. If `data_format="channels_last"`, `inputs` is of shape `(batch_size, ..., channels)` while if `data_format="channels_first"`, `inputs` is of shape `(batch_size, channels, ...)`.
- **dilation_rate**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the dilation rate to use for dilated convolution. If `dilation_rate` is int, then every spatial dimension shares the same `dilation_rate`.

**Returns**

A tensor of rank N+2, the result of the depthwise conv operation.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L2151)

### `dot_product_attention` function

`keras.ops.dot_product_attention(     query, key, value, bias=None, mask=None, scale=None, is_causal=False )`

Scaled dot product attention function.

Computes the attention function on Q (`query`), K (`key`), and V(`value`): `attention(Q, K, V) = softmax(Q * K / sqrt(d)) * V`. If we define `logits` as the output of `Q * K` and the `probs` as the output of `softmax`.

Throughout this function, we utilize the following notation to represent the shape of array: - B: batch size - S: length of the key/value - T: length of the query - N: number of attention heads - H: dimensions of each attention head - K: number of key/value heads - G: number of groups, which equals to `N // K`

**Arguments**

- **query**: The query array with the shape of `(B, T, N, H)`.
- **key**: The key array with the shape of `(B, S, K, H)`. When `K` equals `N`, multi-headed attention (MHA) is performed. Otherwise, grouped query attention (GQA) is performed if `N` is a multiple of `K`. and multi-query attention (MQA) is performed if `K==1` (a special case of GQA).
- **value**: The value array with the same shape of `key`.
- **bias**: Optional bias array to be added to logits. The shape must be broadcastable to `(B, N, T, S)`.
- **mask**: Optional mask array used to filter out logits. It is a boolean mask where `True` indicates the element should take part in attention. For an additive mask, users should pass it to bias. The shape must be broadcastable to `(B, N, T, S)`.
- **scale**: Optional scale for the logits. If `None`, the scale will be set to `1.0 / sqrt(H)`.
- **is_causal**: Whether to apply causal mask.

**Returns**

An array of the attention output with the same shape of `query`.

**Example**

`>>> query = keras.random.normal((2, 4, 8, 16)) >>> key = keras.random.normal((2, 6, 8, 16)) >>> value = keras.random.normal((2, 6, 8, 16)) >>> keras.ops.nn.dot_product_attention(query, key, value).shape (2, 4, 8, 16)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L394)

### `elu` function

`keras.ops.elu(x, alpha=1.0)`

Exponential Linear Unit activation function.

It is defined as:

`f(x) = alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x >= 0`.

**Arguments**

- **x**: Input tensor.
- **alpha**: A scalar, slope of positive section. Defaults to `1.0`.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = np.array([-1., 0., 1.]) >>> x_elu = keras.ops.elu(x) >>> print(x_elu) array([-0.63212055, 0., 1.], shape=(3,), dtype=float64)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L470)

### `gelu` function

`keras.ops.gelu(x, approximate=True)`

Gaussian Error Linear Unit (GELU) activation function.

If `approximate` is `True`, it is defined as: `f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`

Or if `approximate` is `False`, it is defined as: `f(x) = x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, where `P(X) ~ N(0, 1)`.

**Arguments**

- **x**: Input tensor.
- **approximate**: Approximate version of GELU activation. Defaults to `True`.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = np.array([-1., 0., 1.]) >>> x_gelu = keras.ops.gelu(x) >>> print(x_gelu) array([-0.15865525, 0., 0.84134475], shape=(3,), dtype=float64)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L305)

### `hard_sigmoid` function

`keras.ops.hard_sigmoid(x)`

Hard sigmoid activation function.

It is defined as:

`0 if x < -2.5`, `1 if x > 2.5`, `(0.2 * x) + 0.5 if -2.5 <= x <= 2.5`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = np.array([-1., 0., 1.]) >>> x_hard_sigmoid = keras.ops.hard_sigmoid(x) >>> print(x_hard_sigmoid) array([0.3, 0.5, 0.7], shape=(3,), dtype=float64)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L268)

### `leaky_relu` function

`keras.ops.leaky_relu(x, negative_slope=0.2)`

Leaky version of a Rectified Linear Unit activation function.

It allows a small gradient when the unit is not active, it is defined as:

`f(x) = alpha * x for x < 0` or `f(x) = x for x >= 0`.

**Arguments**

- **x**: Input tensor.
- **negative_slope**: Slope of the activation function at x < 0. Defaults to `0.2`.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = np.array([-1., 0., 1.]) >>> x_leaky_relu = keras.ops.leaky_relu(x) >>> print(x_leaky_relu) array([-0.2,  0. ,  1. ], shape=(3,), dtype=float64)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L227)

### `log_sigmoid` function

`keras.ops.log_sigmoid(x)`

Logarithm of the sigmoid activation function.

It is defined as `f(x) = log(1 / (1 + exp(-x)))`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-0.541391, 0.0, 0.50, 5.0]) >>> keras.ops.log_sigmoid(x) array([-1.0000418, -0.6931472, -0.474077, -0.00671535], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L585)

### `log_softmax` function

`keras.ops.log_softmax(x, axis=-1)`

Log-softmax activation function.

It is defined as: `f(x) = x - max(x) - log(sum(exp(x - max(x))))`

**Arguments**

- **x**: Input tensor.
- **axis**: Integer, axis along which the log-softmax is applied. Defaults to `-1`.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = np.array([-1., 0., 1.]) >>> x_log_softmax = keras.ops.log_softmax(x) >>> print(x_log_softmax) array([-2.40760596, -1.40760596, -0.40760596], shape=(3,), dtype=float64)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L668)

### `max_pool` function

`keras.ops.max_pool(     inputs, pool_size, strides=None, padding="valid", data_format=None )`

Max pooling operation.

**Arguments**

- **inputs**: Tensor of rank N+2. `inputs` has shape `(batch_size,) + inputs_spatial_shape + (num_channels,)` if `data_format="channels_last"`, or `(batch_size, num_channels) + inputs_spatial_shape` if `data_format="channels_first"`. Pooling happens over the spatial dimensions only.
- **pool_size**: int or tuple/list of integers of size `len(inputs_spatial_shape)`, specifying the size of the pooling window for each spatial dimension of the input tensor. If `pool_size` is int, then every spatial dimension shares the same `pool_size`.
- **strides**: int or tuple/list of integers of size `len(inputs_spatial_shape)`. The stride of the sliding window for each spatial dimension of the input tensor. If `strides` is int, then every spatial dimension shares the same `strides`.
- **padding**: string, either `"valid"` or `"same"`. `"valid"` means no padding is applied, and `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input when `strides=1`.
- **data_format**: A string, either `"channels_last"` or `"channels_first"`. `data_format` determines the ordering of the dimensions in the inputs. If `data_format="channels_last"`, `inputs` is of shape `(batch_size, ..., channels)` while if `data_format="channels_first"`, `inputs` is of shape `(batch_size, channels, ...)`.

**Returns**

A tensor of rank N+2, the result of the max pooling operation.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1679)

### `moments` function

`keras.ops.moments(x, axes, keepdims=False, synchronized=False)`

Calculates the mean and variance of `x`.

The mean and variance are calculated by aggregating the contents of `x` across `axes`. If `x` is 1-D and `axes = [0]` this is just the mean and variance of a vector.

**Arguments**

- **x**: Input tensor.
- **axes**: A list of axes which to compute mean and variance.
- **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one.
- **synchronized**: Only applicable with the TensorFlow backend. If `True`, synchronizes the global batch statistics (mean and variance) across all devices at each training step in a distributed training strategy. If `False`, each replica uses its own local batch statistics.

**Returns**

A tuple containing two tensors - mean and variance.

**Example**

`>>> x = keras.ops.convert_to_tensor([0, 1, 2, 3, 100], dtype="float32") >>> keras.ops.moments(x, axes=[0]) (array(21.2, dtype=float32), array(1553.3601, dtype=float32))`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1606)

### `multi_hot` function

`keras.ops.multi_hot(     inputs, num_classes=None, axis=-1, dtype=None, sparse=False, **kwargs )`

Encodes integer labels as multi-hot vectors.

This function encodes integer labels as multi-hot vectors, where each label is mapped to a binary value in the resulting vector.

**Arguments**

- **inputs**: Tensor of integer labels to be converted to multi-hot vectors.
- **num_classes**: Integer, the total number of unique classes.
- **axis**: (optional) Axis along which the multi-hot encoding should be added. Defaults to `-1`, which corresponds to the last dimension.
- **dtype**: (optional) The data type of the resulting tensor. Default is backend's float type.
- **sparse**: Whether to return a sparse tensor; for backends that support sparse tensors.

**Returns**

- **Tensor**: The multi-hot encoded tensor.

**Example**

`>>> data = keras.ops.convert_to_tensor([0, 4]) >>> keras.ops.multi_hot(data, num_classes=5) array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1993)

### `normalize` function

`keras.ops.normalize(x, axis=-1, order=2, epsilon=None)`

Normalizes `x` over the specified axis.

It is defined as: `normalize(x) = x / max(norm(x), epsilon)`.

**Arguments**

- **x**: Input tensor.
- **axis**: The axis or axes along which to perform normalization. Default to -1.
- **order**: The exponent value in the norm formulation. Defaults to 2.
- **epsilon**: A lower bound value for the norm. Defaults to `backend.epsilon()`.

**Returns**

The normalized array.

**Example**

`>>> x = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]) >>> x_norm = keras.ops.math.normalize(x) >>> print(x_norm) array([[0.26726124 0.5345225  0.8017837 ]        [0.45584232 0.5698029  0.68376344]], shape=(2, 3), dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1280)

### `one_hot` function

`keras.ops.one_hot(x, num_classes, axis=-1, dtype=None, sparse=False)`

Converts integer tensor `x` into a one-hot tensor.

The one-hot encoding is a representation where each integer value is converted into a binary vector with a length equal to `num_classes`, and the index corresponding to the integer value is marked as 1, while all other indices are marked as 0.

**Arguments**

- **x**: Integer tensor to be encoded. The shape can be arbitrary, but the dtype should be integer.
- **num_classes**: Number of classes for the one-hot encoding.
- **axis**: Axis along which the encoding is performed. `-1` represents the last axis. Defaults to `-1`.
- **dtype**: (Optional) Data type of the output tensor. If not provided, it defaults to the default data type of the backend.
- **sparse**: Whether to return a sparse tensor; for backends that support sparse tensors.

**Returns**

- **Integer tensor**: One-hot encoded tensor with the same shape as `x` except for the specified `axis` dimension, which will have a length of `num_classes`. The dtype of the output tensor is determined by `dtype` or the default data type of the backend.

**Example**

`>>> x = keras.ops.convert_to_tensor([1, 3, 2, 0]) >>> one_hot(x, num_classes=4) array([[0. 1. 0. 0.]        [0. 0. 0. 1.]        [0. 0. 1. 0.]        [1. 0. 0. 0.]], shape=(4, 4), dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L2078)

### `psnr` function

`keras.ops.psnr(x1, x2, max_val)`

Peak Signal-to-Noise Ratio (PSNR) function.

This function computes the Peak Signal-to-Noise Ratio between two signals, `x1` and `x2`. PSNR is a measure of the quality of a reconstructed signal. The higher the PSNR, the closer the reconstructed signal is to the original signal. Note that it can become negative when the signal power is smaller that the noise power.

**Arguments**

- **x1**: The first input signal.
- **x2**: The second input signal. Must have the same shape as `x1`.
- **max_val**: The maximum possible value in the signals.

**Returns**

- **float**: The PSNR value between `x1` and `x2`.

**Examples**

`>>> x1 = keras.random.normal((2, 4, 4, 3)) >>> x2 = keras.random.normal((2, 4, 4, 3)) >>> max_val = 1.0 >>> keras.ops.nn.psnr(x1, x2, max_val) -3.1697404`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L26)

### `relu` function

`keras.ops.relu(x)`

Rectified linear unit activation function.

It is defined as `f(x) = max(0, x)`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x1 = keras.ops.convert_to_tensor([-1.0, 0.0, 1.0, 0.2]) >>> keras.ops.relu(x1) array([0.0, 0.0, 1.0, 0.2], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L57)

### `relu6` function

`keras.ops.relu6(x)`

Rectified linear unit activation function with upper bound of 6.

It is defined as `f(x) = np.clip(x, 0, 6)`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-3.0, -2.0, 0.1, 0.2, 6.0, 8.0]) >>> keras.ops.relu6(x) array([0.0, 0.0, 0.1, 0.2, 6.0, 6.0], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L430)

### `selu` function

`keras.ops.selu(x)`

Scaled Exponential Linear Unit (SELU) activation function.

It is defined as:

`f(x) = scale * alpha * (exp(x) - 1.) for x < 0`, `f(x) = scale * x for x >= 0`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = np.array([-1., 0., 1.]) >>> x_selu = keras.ops.selu(x) >>> print(x_selu) array([-1.11133055, 0., 1.05070098], shape=(3,), dtype=float64)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1052)

### `separable_conv` function

`keras.ops.separable_conv(     inputs,     depthwise_kernel,     pointwise_kernel,     strides=1,     padding="valid",     data_format=None,     dilation_rate=1, )`

General N-D separable convolution.

This ops supports 1D and 2D separable convolution. `separable_conv` is a depthwise conv followed by a pointwise conv.

**Arguments**

- **inputs**: Tensor of rank N+2. `inputs` has shape `(batch_size,) + inputs_spatial_shape + (num_channels,)` if `data_format="channels_last"`, or `(batch_size, num_channels) + inputs_spatial_shape` if `data_format="channels_first"`.
- **depthwise_kernel**: Tensor of rank N+2. `depthwise_kernel` has shape \[kernel_spatial_shape, num_input_channels, num_channels_multiplier\], `num_input_channels` should match the number of channels in `inputs`.
- **pointwise_kernel**: Tensor of rank N+2. `pointwise_kernel` has shape `(*ones_like(kernel_spatial_shape), num_input_channels * num_channels_multiplier, num_output_channels)`.
- **strides**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the strides of the convolution along each spatial dimension. If `strides` is int, then every spatial dimension shares the same `strides`.
- **padding**: string, either `"valid"` or `"same"`. `"valid"` means no padding is applied, and `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input when `strides=1`.
- **data_format**: A string, either `"channels_last"` or `"channels_first"`. `data_format` determines the ordering of the dimensions in the inputs. If `data_format="channels_last"`, `inputs` is of shape `(batch_size, ..., channels)` while if `data_format="channels_first"`, `inputs` is of shape `(batch_size, channels, ...)`.
- **dilation_rate**: int or int tuple/list of `len(inputs_spatial_shape)`, specifying the dilation rate to use for dilated convolution. If `dilation_rate` is int, then every spatial dimension shares the same `dilation_rate`.

**Returns**

A tensor of rank N+2, the result of the depthwise conv operation.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L88)

### `sigmoid` function

`keras.ops.sigmoid(x)`

Sigmoid activation function.

It is defined as `f(x) = 1 / (1 + exp(-x))`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0]) >>> keras.ops.sigmoid(x) array([0.00247262, 0.7310586, 0.5, 0.7310586, 0.9975274], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L185)

### `silu` function

`keras.ops.silu(x)`

Sigmoid Linear Unit (SiLU) activation function, also known as Swish.

The SiLU activation function is computed by the sigmoid function multiplied by its input. It is defined as `f(x) = x * sigmoid(x)`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0]) >>> keras.ops.sigmoid(x) array([0.00247262, 0.7310586, 0.5, 0.7310586, 0.9975274], dtype=float32) >>> keras.ops.silu(x) array([-0.0148357, 0.7310586, 0.0, 0.7310586, 5.9851646], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L345)

### `hard_silu` function

`keras.ops.hard_silu(x)`

Hard SiLU activation function, also known as Hard Swish.

It is defined as:

- `0` if `if x < -3`
- `x` if `x > 3`
- `x * (x + 3) / 6` if `-3 <= x <= 3`

It's a faster, piecewise linear approximation of the silu activation.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-3.0, -1.0, 0.0, 1.0, 3.0]) >>> keras.ops.hard_silu(x) array([-0.0, -0.3333333, 0.0, 0.6666667, 3.0], shape=(5,), dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L513)

### `softmax` function

`keras.ops.softmax(x, axis=-1)`

Softmax activation function.

The elements of the output vector lie within the range `(0, 1)`, and their total sum is exactly 1 (excluding the floating point rounding error).

Each vector is processed independently. The `axis` argument specifies the axis along which the function is applied within the input.

It is defined as: `f(x) = exp(x) / sum(exp(x))`

**Arguments**

- **x**: Input tensor.
- **axis**: Integer, axis along which the softmax is applied.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = np.array([-1., 0., 1.]) >>> x_softmax = keras.ops.softmax(x) >>> print(x_softmax) array([0.09003057, 0.24472847, 0.66524096], shape=(3,), dtype=float64)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L120)

### `softplus` function

`keras.ops.softplus(x)`

Softplus activation function.

It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural logarithm and `exp` is the exponential function.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-0.555, 0.0, 0.555]) >>> keras.ops.softplus(x) array([0.45366603, 0.6931472, 1.008666], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L153)

### `softsign` function

`keras.ops.softsign(x)`

Softsign activation function.

It is defined as `f(x) = x / (abs(x) + 1)`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-0.100, -10.0, 1.0, 0.0, 100.0]) >>> keras.ops.softsign(x) Array([-0.09090909, -0.90909094, 0.5, 0.0, 0.990099], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L1509)

### `sparse_categorical_crossentropy` function

`keras.ops.sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1)`

Computes sparse categorical cross-entropy loss.

The sparse categorical cross-entropy loss is similar to categorical cross-entropy, but it is used when the target tensor contains integer class labels instead of one-hot encoded vectors. It measures the dissimilarity between the target and output probabilities or logits.

**Arguments**

- **target**: The target tensor representing the true class labels as integers. Its shape should match the shape of the `output` tensor except for the last dimension.
- **output**: The output tensor representing the predicted probabilities or logits. Its shape should match the shape of the `target` tensor except for the last dimension.
- **from_logits**: (optional) Whether `output` is a tensor of logits or probabilities. Set it to `True` if `output` represents logits; otherwise, set it to `False` if `output` represents probabilities. Defaults to `False`.
- **axis**: (optional) The axis along which the sparse categorical cross-entropy is computed. Defaults to `-1`, which corresponds to the last dimension of the tensors.

**Returns**

- **Integer tensor**: The computed sparse categorical cross-entropy loss between `target` and `output`.

**Example**

`>>> target = keras.ops.convert_to_tensor([0, 1, 2], dtype=int32) >>> output = keras.ops.convert_to_tensor( ... [[0.9, 0.05, 0.05], ...  [0.1, 0.8, 0.1], ...  [0.2, 0.3, 0.5]]) >>> sparse_categorical_crossentropy(target, output) array([0.10536056 0.22314355 0.6931472 ], shape=(3,), dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L185)

### `silu` function

`keras.ops.swish(x)`

Sigmoid Linear Unit (SiLU) activation function, also known as Swish.

The SiLU activation function is computed by the sigmoid function multiplied by its input. It is defined as `f(x) = x * sigmoid(x)`.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0]) >>> keras.ops.sigmoid(x) array([0.00247262, 0.7310586, 0.5, 0.7310586, 0.9975274], dtype=float32) >>> keras.ops.silu(x) array([-0.0148357, 0.7310586, 0.0, 0.7310586, 5.9851646], dtype=float32)`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/nn.py#L345)

### `hard_silu` function

`keras.ops.hard_swish(x)`

Hard SiLU activation function, also known as Hard Swish.

It is defined as:

- `0` if `if x < -3`
- `x` if `x > 3`
- `x * (x + 3) / 6` if `-3 <= x <= 3`

It's a faster, piecewise linear approximation of the silu activation.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-3.0, -1.0, 0.0, 1.0, 3.0]) >>> keras.ops.hard_silu(x) array([-0.0, -0.3333333, 0.0, 0.6666667, 3.0], shape=(5,), dtype=float32)`

---
