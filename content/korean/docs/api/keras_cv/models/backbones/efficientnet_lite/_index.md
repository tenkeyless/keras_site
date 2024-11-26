---
title: EfficientNet Lite backbones
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_lite/efficientnet_lite_backbone.py#L39" >}}

### `EfficientNetLiteBackbone` class

```python
keras_cv.models.EfficientNetLiteBackbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="relu6",
    **kwargs
)
```

Instantiates the EfficientNetLite architecture using given scaling
coefficients.

**Reference**

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  (ICML 2019)
- [Based on the original EfficientNet Lite's](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)

**Arguments**

- **include_rescaling**: whether to rescale the inputs. If set to True,
  inputs will be passed through a `Rescaling(1/255.0)` layer.
- **width_coefficient**: float, scaling coefficient for network width.
- **depth_coefficient**: float, scaling coefficient for network depth.
- **dropout_rate**: float, dropout rate before final classifier layer.
- **drop_connect_rate**: float, dropout rate at skip connections. The
  default value is set to 0.2.
- **depth_divisor**: integer, a unit of network width. The default value
  is set to 8.
- **activation**: activation function.
- **input_shape**: optional shape tuple,
  It should have exactly 3 inputs channels.
- **input_tensor**: optional Keras tensor (i.e. output of `keras.layers.Input()`)
  to use as image input for the model.

**Example**

```python
# Construct an EfficientNetLite from a preset:
efficientnet = models.EfficientNetLiteBackbone.from_preset(
    "efficientnetlite_b0"
)
images = np.ones((1, 256, 256, 3))
outputs = efficientnet.predict(images)
# Alternatively, you can also customize the EfficientNetLite architecture:
model = EfficientNetLiteBackbone(
    stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
    stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
    stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
    stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
    stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
    stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
    width_coefficient=1.0,
    depth_coefficient=1.0,
    include_rescaling=False,
)
images = np.ones((1, 256, 256, 3))
outputs = model.predict(images)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/backbone.py#L132" >}}

### `from_preset` method

```python
EfficientNetLiteBackbone.from_preset()
```

Instantiate EfficientNetLiteBackbone model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "efficientnetlite_b0", "efficientnetlite_b1", "efficientnetlite_b2", "efficientnetlite_b3", "efficientnetlite_b4".
  If looking for a preset with pretrained weights, choose one of
  "".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.

**Examples**

```python
# Load architecture and weights from preset
model = keras_cv.models.EfficientNetLiteBackbone.from_preset(
    "",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.EfficientNetLiteBackbone.from_preset(
    "",
    load_weights=False,
```

| Preset name         | Parameters | Description                                                                                                                                |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| efficientnetlite_b0 | 3.41M      | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.0`. |
| efficientnetlite_b1 | 4.19M      | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.0` and `depth_coefficient=1.1`. |
| efficientnetlite_b2 | 4.87M      | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.1` and `depth_coefficient=1.2`. |
| efficientnetlite_b3 | 6.99M      | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.2` and `depth_coefficient=1.4`. |
| efficientnetlite_b4 | 11.84M     | EfficientNet B-style architecture with 7 convolutional blocks. This B-style model has `width_coefficient=1.4` and `depth_coefficient=1.8`. |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_lite/efficientnet_lite_aliases.py#L45" >}}

### `EfficientNetLiteB0Backbone` class

```python
keras_cv.models.EfficientNetLiteB0Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="relu6",
    **kwargs
)
```

Instantiates the EfficientNetLiteB0 architecture.

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

**Example**

```python
input_data = np.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = EfficientNetLiteB0Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_lite/efficientnet_lite_aliases.py#L78" >}}

### `EfficientNetLiteB1Backbone` class

```python
keras_cv.models.EfficientNetLiteB1Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="relu6",
    **kwargs
)
```

Instantiates the EfficientNetLiteB1 architecture.

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

**Example**

```python
input_data = np.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = EfficientNetLiteB1Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_lite/efficientnet_lite_aliases.py#L111" >}}

### `EfficientNetLiteB2Backbone` class

```python
keras_cv.models.EfficientNetLiteB2Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="relu6",
    **kwargs
)
```

Instantiates the EfficientNetLiteB2 architecture.

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

**Example**

```python
input_data = np.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = EfficientNetLiteB2Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_lite/efficientnet_lite_aliases.py#L144" >}}

### `EfficientNetLiteB3Backbone` class

```python
keras_cv.models.EfficientNetLiteB3Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="relu6",
    **kwargs
)
```

Instantiates the EfficientNetLiteB3 architecture.

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

**Example**

```python
input_data = np.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = EfficientNetLiteB3Backbone()
output = model(input_data)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/backbones/efficientnet_lite/efficientnet_lite_aliases.py#L177" >}}

### `EfficientNetLiteB4Backbone` class

```python
keras_cv.models.EfficientNetLiteB4Backbone(
    include_rescaling,
    width_coefficient,
    depth_coefficient,
    stackwise_kernel_sizes,
    stackwise_num_repeats,
    stackwise_input_filters,
    stackwise_output_filters,
    stackwise_expansion_ratios,
    stackwise_strides,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    input_shape=(None, None, 3),
    input_tensor=None,
    activation="relu6",
    **kwargs
)
```

Instantiates the EfficientNetLiteB4 architecture.

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

**Example**

```python
input_data = np.ones(shape=(8, 224, 224, 3))
# Randomly initialized backbone
model = EfficientNetLiteB4Backbone()
output = model(input_data)
```
