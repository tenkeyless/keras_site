---
title: EfficientNetV2 models
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_backbone.py#L32" >}}

### `EfficientNetV2Backbone` class

```python
keras_cv.models.EfficientNetV2Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **width_coefficient**: float, scaling coefficient for network width.
- **depth_coefficient**: float, scaling coefficient for network depth.
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
- **stackwise_squeeze_and_excite_ratios**: list of ints, the squeeze and
  excite ratios passed to the squeeze and excitation blocks.
- **stackwise_strides**: list of ints, stackwise_strides for each conv block.
- **stackwise_conv_types**: list of strings. Each value is either 'unfused'
  or 'fused' depending on the desired blocks. FusedMBConvBlock is
  similar to MBConvBlock, but instead of using a depthwise convolution
  and a 1x1 output convolution blocks fused blocks use a single 3x3
  convolution block.
- **skip_connection_dropout**: float, dropout rate at skip connections.
- **depth_divisor**: integer, a unit of network width.
- **min_depth**: integer, minimum number of filters.
- **activation**: activation function to use between each convolutional layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `keras.layers.Input()`)
  to use as image input for the model.

**Example**

```python
# Construct an EfficientNetV2 from a preset:
efficientnet = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_s"
)
images = tf.ones((1, 256, 256, 3))
outputs = efficientnet.predict(images)
# Alternatively, you can also customize the EfficientNetV2 architecture:
model = EfficientNetV2Backbone(
    stackwise_kernel_sizes=[3, 3, 3, 3, 3, 3],
    stackwise_num_repeats=[2, 4, 4, 6, 9, 15],
    stackwise_input_filters=[24, 24, 48, 64, 128, 160],
    stackwise_output_filters=[24, 48, 64, 128, 160, 256],
    stackwise_expansion_ratios=[1, 4, 4, 4, 6, 6],
    stackwise_squeeze_and_excite_ratios=[0.0, 0.0, 0, 0.25, 0.25, 0.25],
    stackwise_strides=[1, 2, 2, 2, 1, 2],
    stackwise_conv_types=[
        "fused",
        "fused",
        "fused",
        "unfused",
        "unfused",
        "unfused",
    ],
    width_coefficient=1.0,
    depth_coefficient=1.0,
    include_rescaling=False,
)
images = tf.ones((1, 256, 256, 3))
outputs = efficientnet.predict(images)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
EfficientNetV2Backbone.from_preset()
```

Instantiate EfficientNetV2Backbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_b0", "efficientnetv2_b1", "efficientnetv2_b2", "efficientnetv2_b3", "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet".
  If looking for a preset with pretrained weights, choose one of
  "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_s_imagenet",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_s_imagenet",
    load_weights=False,
```

| Preset name                | Parameters | Description                                                                                                                                                                                                                                                                                                           |
| -------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| efficientnetv2_s           | 20.33M     | EfficientNet architecture with 6 convolutional blocks.                                                                                                                                                                                                                                                                |
| efficientnetv2_m           | 53.15M     | EfficientNet architecture with 7 convolutional blocks.                                                                                                                                                                                                                                                                |
| efficientnetv2_l           | 117.75M    | EfficientNet architecture with 7 convolutional blocks, but more filters the in `efficientnetv2_m`.                                                                                                                                                                                                                    |
| efficientnetv2_b0          | 5.92M      | EfficientNet B-style architecture with 6 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.0`.                                                                                                                                                                            |
| efficientnetv2_b1          | 6.93M      | EfficientNet B-style architecture with 6 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.1`.                                                                                                                                                                            |
| efficientnetv2_b2          | 8.77M      | EfficientNet B-style architecture with 6 convolutional blocks. This B-style model has `width_coefficient=1.1` and `depth_coefficient=1.2`.                                                                                                                                                                            |
| efficientnetv2_b3          | 12.93M     | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.2` and `depth_coefficient=1.4`.                                                                                                                                                                            |
| efficientnetv2_s_imagenet  | 20.33M     | EfficientNet architecture with 6 convolutional blocks. Weights are initialized to pretrained imagenet classification weights.Published weights are capable of scoring 83.9%top 1 accuracy and 96.7% top 5 accuracy on imagenet.                                                                                       |
| efficientnetv2_b0_imagenet | 5.92M      | EfficientNet B-style architecture with 6 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.0`. Weights are initialized to pretrained imagenet classification weights. Published weights are capable of scoring 77.1% top 1 accuracy and 93.3% top 5 accuracy on imagenet. |
| efficientnetv2_b1_imagenet | 6.93M      | EfficientNet B-style architecture with 6 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.1`. Weights are initialized to pretrained imagenet classification weights.Published weights are capable of scoring 79.1% top 1 accuracy and 94.4% top 5 accuracy on imagenet.  |
| efficientnetv2_b2_imagenet | 8.77M      | EfficientNet B-style architecture with 6 convolutional blocks. This B-style model has `width_coefficient=1.1` and `depth_coefficient=1.2`. Weights are initialized to pretrained imagenet classification weights.Published weights are capable of scoring 80.1% top 1 accuracy and 94.9% top 5 accuracy on imagenet.  |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_aliases.py#L138" >}}

### `EfficientNetV2B0Backbone` class

```python
keras_cv.models.EfficientNetV2B0Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2B0 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_aliases.py#L173" >}}

### `EfficientNetV2B1Backbone` class

```python
keras_cv.models.EfficientNetV2B1Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2B1 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_aliases.py#L208" >}}

### `EfficientNetV2B2Backbone` class

```python
keras_cv.models.EfficientNetV2B2Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2B2 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_aliases.py#L243" >}}

### `EfficientNetV2B3Backbone` class

```python
keras_cv.models.EfficientNetV2B3Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2B3 architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_aliases.py#L41" >}}

### `EfficientNetV2SBackbone` class

```python
keras_cv.models.EfficientNetV2SBackbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2S architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_aliases.py#L76" >}}

### `EfficientNetV2MBackbone` class

```python
keras_cv.models.EfficientNetV2MBackbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2M architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_v2/efficientnet_v2_aliases.py#L107" >}}

### `EfficientNetV2LBackbone` class

```python
keras_cv.models.EfficientNetV2LBackbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_squeeze_and_excite_ratios,
    stackwise_strides,
    stackwise_conv_types,
    skip_connection_dropout=0.2,
    depth_divisor=8,
    min_depth=8,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs
)
```

Instantiates the EfficientNetV2L architecture.

**Reference**

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  (ICML 2021)

**Arguments**

- **include_rescaling**: bool, whether to rescale the inputs. If set
  to `True`, inputs will be passed through a `Rescaling(1/255.0)`
  layer.
- **input_shape**: optional shape tuple, defaults to (None, None, 3).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
