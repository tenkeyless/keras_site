---
title: Conv1D layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/convolutional/conv1d.py#L6" >}}

### `Conv1D` class

```python
keras.layers.Conv1D(
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

1D convolution layer (e.g. temporal convolution).

This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If `use_bias` is True, a bias vector is created and added to the outputs. Finally, if `activation` is not `None`, it is applied to the outputs as well.

**Arguments**

- **filters**: int, the dimension of the output space (the number of filters in the convolution).
- **kernel_size**: int or tuple/list of 1 integer, specifying the size of the convolution window.
- **strides**: int or tuple/list of 1 integer, specifying the stride length of the convolution. `strides > 1` is incompatible with `dilation_rate > 1`.
- **padding**: string, `"valid"`, `"same"` or `"causal"`(case-insensitive). `"valid"` means no padding. `"same"` results in padding evenly to the left/right or up/down of the input. When `padding="same"` and `strides=1`, the output has the same size as the input. `"causal"` results in causal(dilated) convolutions, e.g. `output[t]` does not depend on`input[t+1:]`. Useful when modeling temporal data where the model should not violate the temporal order. See [WaveNet: A Generative Model for Raw Audio, section2.1](https://arxiv.org/abs/1609.03499).
- **data_format**: string, either `"channels_last"` or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, steps, features)` while `"channels_first"` corresponds to inputs with shape `(batch, features, steps)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. If you never set it, then it will be `"channels_last"`.
- **dilation_rate**: int or tuple/list of 1 integers, specifying the dilation rate to use for dilated convolution.
- **groups**: A positive int specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with `filters // groups` filters. The output is the concatenation of all the `groups` results along the channel axis. Input channels and `filters` must both be divisible by `groups`.
- **activation**: Activation function. If `None`, no activation is applied.
- **use_bias**: bool, if `True`, bias will be added to the output.
- **kernel_initializer**: Initializer for the convolution kernel. If `None`, the default initializer (`"glorot_uniform"`) will be used.
- **bias_initializer**: Initializer for the bias vector. If `None`, the default initializer (`"zeros"`) will be used.
- **kernel_regularizer**: Optional regularizer for the convolution kernel.
- **bias_regularizer**: Optional regularizer for the bias vector.
- **activity_regularizer**: Optional regularizer function for the output.
- **kernel_constraint**: Optional projection function to be applied to the kernel after being updated by an `Optimizer` (e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected variable and must return the projected variable (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
- **bias_constraint**: Optional projection function to be applied to the bias after being updated by an `Optimizer`.

**Input shape**

- If `data_format="channels_last"`: A 3D tensor with shape: `(batch_shape, steps, channels)`
- If `data_format="channels_first"`: A 3D tensor with shape: `(batch_shape, channels, steps)`

**Output shape**

- If `data_format="channels_last"`: A 3D tensor with shape: `(batch_shape, new_steps, filters)`
- If `data_format="channels_first"`: A 3D tensor with shape: `(batch_shape, filters, new_steps)`

**Returns**

A 3D tensor representing `activation(conv1d(inputs, kernel) + bias)`.

**Raises**

- **ValueError**: when both `strides > 1` and `dilation_rate > 1`.

**Example**

```console
>>> # The inputs are 128-length vectors with 10 timesteps, and the
>>> # batch size is 4.
>>> x = np.random.rand(4, 10, 128)
>>> y = keras.layers.Conv1D(32, 3, activation='relu')(x)
>>> print(y.shape)
(4, 8, 32)
```
