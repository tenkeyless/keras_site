---
title: EfficientNetV1 models
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_backbone.py#L27" >}}

### `EfficientNetV1Backbone` class

```python
keras_cv.models.EfficientNetV1Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)
- [Based on the original keras.applications EfficientNet](https://github.com/keras-team/keras/blob/master/keras/applications/efficientnet.py)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set to
  True, inputs will be passed through a `Rescaling(1/255.0)` layer.
- **width_coefficient**: float, scaling coefficient for network width.
- **depth_coefficient**: float, scaling coefficient for network depth.
- **dropout_rate**: float, dropout rate before final classifier layer.
- **drop_connect_rate**: float, dropout rate at skip connections. The default
  value is set to 0.2.
- **depth_divisor**: integer, a unit of network width. The default value is
  set to 8.
- **activation**: activation function to use between each convolutional layer.
- **input_shape**: optional shape tuple, it should have exactly 3 input
  channels.
- **input_tensor**: optional Keras tensor (i.e. output of `keras.keras.layers.Input()`) to
  use as image input for the model.
- **stackwise_kernel_sizes**: list of ints, the kernel sizes used for each
  conv block.
- **stackwise_num_repeats**: list of ints, number of times to repeat each
  conv block.
- **stackwise_input_filters**: list of ints, number of input filters for
  each conv block.
- **stackwise_output_filters**: list of ints, number of output filters for
  each stack in the conv blocks model.
- **stackwise_expansion_ratios**: list of floats, expand ratio passed to the
  squeeze and excitation blocks.
- **stackwise_strides**: list of ints, stackwise_strides for each conv block.
- **stackwise_squeeze_and_excite_ratios**: list of ints, the squeeze and
  excite ratios passed to the squeeze and excitation blocks.

**Example**

```python
# Construct an EfficientNetV1 from a preset:
efficientnet = keras_cv.models.EfficientNetV1Backbone.from_preset(
    "efficientnetv1_b0"
)
images = np.ones((1, 256, 256, 3))
outputs = efficientnet.predict(images)
# Alternatively, you can also customize the EfficientNetV1 architecture:
model = EfficientNetV1Backbone(
    stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
    stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
    stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
    stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
    stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
    stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
    stackwise_squeeze_and_excite_ratios=[
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    ],
    width_coefficient=1.0,
    depth_coefficient=1.0,
    include_rescaling=False,
)
images = np.ones((1, 256, 256, 3))
outputs = efficientnet.predict(images)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
EfficientNetV1Backbone.from_preset()
```

Instantiate EfficientNetV1Backbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "efficientnetv1_b0", "efficientnetv1_b1", "efficientnetv1_b2", "efficientnetv1_b3", "efficientnetv1_b4", "efficientnetv1_b5", "efficientnetv1_b6", "efficientnetv1_b7".
  If looking for a preset with pretrained weights, choose one of
  "".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.EfficientNetV1Backbone.from_preset(
    "",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.EfficientNetV1Backbone.from_preset(
    "",
    load_weights=False,
```

| Preset name       | Parameters | Description                                                                                                                                |
| ----------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| efficientnetv1_b0 | 4.05M      | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.0`. |
| efficientnetv1_b1 | 6.58M      | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.1`. |
| efficientnetv1_b2 | 7.77M      | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.1` and `depth_coefficient=1.2`. |
| efficientnetv1_b3 | 10.79M     | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.2` and `depth_coefficient=1.4`. |
| efficientnetv1_b4 | 17.68M     | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.4` and `depth_coefficient=1.8`. |
| efficientnetv1_b5 | 28.52M     | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.6` and `depth_coefficient=2.2`. |
| efficientnetv1_b6 | 40.97M     | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.8` and `depth_coefficient=2.6`. |
| efficientnetv1_b7 | 64.11M     | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=2.0` and `depth_coefficient=3.1`. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L37" >}}

### `EfficientNetV1B0Backbone` class

```python
keras_cv.models.EfficientNetV1B0Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B0 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L68" >}}

### `EfficientNetV1B1Backbone` class

```python
keras_cv.models.EfficientNetV1B1Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B1 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L99" >}}

### `EfficientNetV1B2Backbone` class

```python
keras_cv.models.EfficientNetV1B2Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B2 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L130" >}}

### `EfficientNetV1B3Backbone` class

```python
keras_cv.models.EfficientNetV1B3Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B3 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L161" >}}

### `EfficientNetV1B4Backbone` class

```python
keras_cv.models.EfficientNetV1B4Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B4 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L192" >}}

### `EfficientNetV1B5Backbone` class

```python
keras_cv.models.EfficientNetV1B5Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B5 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L223" >}}

### `EfficientNetV1B6Backbone` class

```python
keras_cv.models.EfficientNetV1B6Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B6 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v1/efficientnet_v1_aliases.py#L254" >}}

### `EfficientNetV1B7Backbone` class

```python
keras_cv.models.EfficientNetV1B7Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    stackwise_squeeze_and_excite_ratios,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="swish",
    **kwargs
)
```

Instantiates the EfficientNetV1B7 architecture.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
