---
title: CategoryEncoding layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/category_encoding.py#L8" >}}

### `CategoryEncoding` class

```python
keras.layers.CategoryEncoding(
    num_tokens=None, output_mode="multi_hot", sparse=False, **kwargs
)
```

A preprocessing layer which encodes integer features.

This layer provides options for condensing data into a categorical encoding when the total number of tokens are known in advance. It accepts integer values as inputs, and it outputs a dense or sparse representation of those inputs. For integer inputs where the total number of tokens is not known, use [`keras.layers.IntegerLookup`]({{< relref "/docs/api/layers/preprocessing_layers/categorical/integer_lookup#integerlookup-class" >}}) instead.

**Note:** This layer is safe to use inside a [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipeline (independently of which backend you're using).

**Examples**

**One-hot encoding data**

```console
>>> layer = keras.layers.CategoryEncoding(
...           num_tokens=4, output_mode="one_hot")
>>> layer([3, 2, 0, 1])
array([[0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]]>
```

**Multi-hot encoding data**

```console
>>> layer = keras.layers.CategoryEncoding(
...           num_tokens=4, output_mode="one_hot")
>>> layer([3, 2, 0, 1])
array([[0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]]>
```

**Using weighted inputs in `"count"` mode**

```console
>>> layer = keras.layers.CategoryEncoding(
...           num_tokens=4, output_mode="one_hot")
>>> layer([3, 2, 0, 1])
array([[0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]]>
```

**Arguments**

- **num_tokens**: The total number of tokens the layer should support. All inputs to the layer must integers in the range `0 <= value < num_tokens`, or an error will be thrown.
- **output_mode**: Specification for the output of the layer. Values can be `"one_hot"`, `"multi_hot"` or `"count"`, configuring the layer as follows: - `"one_hot"`: Encodes each individual element in the input into an array of `num_tokens` size, containing a 1 at the element index. If the last dimension is size 1, will encode on that dimension. If the last dimension is not size 1, will append a new dimension for the encoded output. - `"multi_hot"`: Encodes each sample in the input into a single array of `num_tokens` size, containing a 1 for each vocabulary term present in the sample. Treats the last dimension as the sample dimension, if input shape is `(..., sample_length)`, output shape will be `(..., num_tokens)`. - `"count"`: Like `"multi_hot"`, but the int array contains a count of the number of times the token at that index appeared in the sample. For all output modes, currently only output up to rank 2 is supported. Defaults to `"multi_hot"`.
- **sparse**: Whether to return a sparse tensor; for backends that support sparse tensors.

**Call arguments**

- **inputs**: A 1D or 2D tensor of integer inputs.
- **count_weights**: A tensor in the same shape as `inputs` indicating the weight for each sample value when summing up in `count` mode. Not used in `"multi_hot"` or `"one_hot"` modes.
