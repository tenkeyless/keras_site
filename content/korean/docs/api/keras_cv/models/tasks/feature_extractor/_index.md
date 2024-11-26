---
title: CLIP Feature extractor
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_model.py#L72" >}}

### `CLIP` class

```python
keras_cv.models.CLIP(
    embed_dim=512,
    image_resolution=224,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=32,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
    **kwargs
)
```

CLIP implements the Contrastive Language-Image Pretraining (CLIP)
architecture, which enables joint learning of visual and textual
representations for various downstream tasks. The deafult base model
achitecture will be set to clip-vit-base-patch32.

**Arguments**

- **embed_dim (int)**: The dimensionality of the joint embedding space for
  images and texts.
- **image_resolution (int)**: The resolution of the input images (both height
  and width).
- **vision_layers (int)**: The number of layers in the vision (image) encoder.
  vision_width (int): The width of the hidden layers in the vision
  encoder.
- **vision_patch_size (int)**: The size of each square patch in the input
  images.
- **context_length (int)**: The maximum length of the contextualized text
  sequences.
- **vocab_size (int)**: The size of the vocabulary for tokenization.
- **transformer_width (int)**: The width of the hidden layers in the
  transformer-based text encoder.
- **transformer_heads (int)**: The number of attention heads in the
  transformer-based text encoder.
- **transformer_layers (int)**: The number of layers in the transformer-based
  text encoder.

**Example**

```python
processor = CLIPProcessor(
    input_resolution=224,
    "path_to_vocab.json",
    "path_to_merges.txt"
)
processed_image = processor.process_images(["cat.jpg"])
tokens = processor(
    ["mountains", "cat on tortoise", "two cats"]
)
model = CLIP.from_preset("clip-vit-base-patch16")
image_logits, text_logits = model(
    {
        "images": processed_image,
        "token_ids": tokens["token_ids"],
        "padding_mask": tokens["padding_mask"],
    }
)
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/task.py#L183" >}}

### `from_preset` method

```python
CLIP.from_preset()
```

Instantiate CLIP model from preset config and weights.

**Arguments**

- **preset**: string. Must be one of "clip-vit-base-patch16", "clip-vit-base-patch32", "clip-vit-large-patch14", "clip-vit-large-patch14-336".
  If looking for a preset with pretrained weights, choose one of
  "clip-vit-base-patch16", "clip-vit-base-patch32", "clip-vit-large-patch14", "clip-vit-large-patch14-336".
- **load_weights**: Whether to load pre-trained weights into model.
  Defaults to `None`, which follows whether the preset has
  pretrained weights available.
- **input_shape** : input shape that will be passed to backbone
  initialization, Defaults to `None`.If `None`, the preset
  value will be used.

**Example**

```python
# Load architecture and weights from preset
model = keras_cv.models.CLIP.from_preset(
    "clip-vit-base-patch16",
)
# Load randomly initialized model from preset architecture with weights
model = keras_cv.models.CLIP.from_preset(
    "clip-vit-base-patch16",
    load_weights=False,
```

| Preset name                | Parameters | Description                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| clip-vit-base-patch16      | 149.62M    | The model uses a ViT-B/16 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss. The model uses a patch size of 16 and input images of size (224, 224) |
| clip-vit-base-patch32      | 151.28M    | The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.The model uses a patch size of 32 and input images of size (224, 224)  |
| clip-vit-large-patch14     | 427.62M    | The model uses a ViT-L/14 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.The model uses a patch size of 14 and input images of size (224, 224)  |
| clip-vit-large-patch14-336 | 427.94M    | The model uses a ViT-L/14 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.The model uses a patch size of 14 and input images of size (336, 336)  |

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_encoder.py#L171" >}}

### `CLIPAttention` class

```python
keras_cv.models.feature_extractor.CLIPAttention(
    proj_dim, num_heads, num_hidden_layers, dropout=0.0, **kwargs
)
```

Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py # noqa: E501

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_encoder.py#L119" >}}

### `CLIPEncoder` class

```python
keras_cv.models.feature_extractor.CLIPEncoder(width, num_layers, heads, **kwargs)
```

This is the class from which all layers inherit.

A layer is a callable object that takes as input one or more tensors and
that outputs one or more tensors. It involves _computation_, defined
in the `call()` method, and a _state_ (weight variables). State can be
created:

- in `__init__()`, for instance via `self.add_weight()`;
- in the optional `build()` method, which is invoked by the first
  `__call__()` to the layer, and supplies the shape(s) of the input(s),
  which may not have been known at initialization time.

Layers are recursively composable: If you assign a Layer instance as an
attribute of another Layer, the outer layer will start tracking the weights
created by the inner layer. Nested layers should be instantiated in the
`__init__()` method or `build()` method.

Users will just instantiate a layer and then treat it as a callable.

**Arguments**

- **trainable**: Boolean, whether the layer's variables should be trainable.
- **name**: String name of the layer.
- **dtype**: The dtype of the layer's computations and weights. Can also be a
  `keras.DTypePolicy`,
  which allows the computation and
  weight dtype to differ. Defaults to `None`. `None` means to use
  `keras.config.dtype_policy()`,
  which is a `float32` policy unless set to different value
  (via `keras.config.set_dtype_policy()`).

**Attributes**

- **name**: The name of the layer (string).
- **dtype**: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
- **variable_dtype**: Dtype of the layer's weights.
- **compute_dtype**: The dtype of the layer's computations.
  Layers automatically cast inputs to this dtype, which causes
  the computations and output to also be in this dtype.
  When mixed precision is used with a
  `keras.DTypePolicy`, this will be different
  than `variable_dtype`.
- **trainable_weights**: List of variables to be included in backprop.
- **non_trainable_weights**: List of variables that should not be
  included in backprop.
- **weights**: The concatenation of the lists trainable_weights and
  non_trainable_weights (in this order).
- **trainable**: Whether the layer should be trained (boolean), i.e.
  whether its potentially-trainable weights should be returned
  as part of `layer.trainable_weights`.
- **input_spec**: Optional (list of) `InputSpec` object(s) specifying the
  constraints on inputs that can be accepted by the layer.

We recommend that descendants of `Layer` implement the following methods:

- `__init__()`: Defines custom layer attributes, and creates layer weights
  that do not depend on input shapes, using `add_weight()`,
  or other state.
- `build(self, input_shape)`: This method can be used to create weights that
  depend on the shape(s) of the input(s), using `add_weight()`, or other
  state. `__call__()` will automatically build the layer
  (if it has not been built yet) by calling `build()`.
- `call(self, *args, **kwargs)`: Called in `__call__` after making
  sure `build()` has been called. `call()` performs the logic of applying
  the layer to the input arguments.
  Two reserved keyword arguments you can optionally use in `call()` are:
  1. `training` (boolean, whether the call is in inference mode or
     training mode).
  2. `mask` (boolean tensor encoding masked timesteps in the input,
     used e.g. in RNN layers).
     A typical signature for this method is `call(self, inputs)`, and user
     could optionally add `training` and `mask` if the layer need them.
- `get_config(self)`: Returns a dictionary containing the configuration
  used to initialize this layer. If the keys differ from the arguments
  in `__init__()`, then override `from_config(self)` as well.
  This method is used when saving
  the layer or a model that contains this layer.

**Examples**

Here's a basic example: a layer with two variables, `w` and `b`,
that returns `y = w . x + b`.
It shows how to implement `build()` and `call()`.
Variables set as attributes of a layer are tracked as weights
of the layers (in `layer.weights`).

```python
class SimpleDense(Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units
    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
    # Defines the computation
    def call(self, inputs):
        return ops.matmul(inputs, self.kernel) + self.bias
# Instantiates the layer.
linear_layer = SimpleDense(4)
# This will also call `build(input_shape)` and create the weights.
y = linear_layer(ops.ones((2, 2)))
assert len(linear_layer.weights) == 2
# These weights are trainable, so they're listed in `trainable_weights`:
assert len(linear_layer.trainable_weights) == 2
```

Besides trainable weights, updated via backpropagation during training,
layers can also have non-trainable weights. These weights are meant to
be updated manually during `call()`. Here's a example layer that computes
the running sum of its inputs:

```python
class ComputeSum(Layer):
  def __init__(self, input_dim):
      super(ComputeSum, self).__init__()
      # Create a non-trainable weight.
      self.total = self.add_weight(
        shape=(),
        initializer="zeros",
        trainable=False,
        name="total",
      )
  def call(self, inputs):
      self.total.assign(self.total + ops.sum(inputs))
      return self.total
my_sum = ComputeSum(2)
x = ops.ones((2, 2))
y = my_sum(x)
assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_image_model.py#L100" >}}

### `CLIPImageEncoder` class

```python
keras_cv.models.feature_extractor.CLIPImageEncoder(
    input_resolution, patch_size, width, num_layers, heads, output_dim, **kwargs
)
```

A model grouping layers into an object with training/inference features.

There are three ways to instantiate a `Model`:

## With the "Functional API"

You start from `Input`,
you chain layer calls to specify the model's forward pass,
and finally, you create your model from inputs and outputs:

```python
inputs = keras.Input(shape=(37,))
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

Note: Only dicts, lists, and tuples of input tensors are supported. Nested
inputs are not supported (e.g. lists of list or dicts of dict).

A new Functional API model can also be created by using the
intermediate tensors. This enables you to quickly extract sub-components
of the model.

**Example**

```python
inputs = keras.Input(shape=(None, None, 3))
processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
pooling = keras.layers.GlobalAveragePooling2D()(conv)
feature = keras.layers.Dense(10)(pooling)
full_model = keras.Model(inputs, feature)
backbone = keras.Model(processed, conv)
activations = keras.Model(conv, feature)
```

Note that the `backbone` and `activations` models are not
created with [`keras.Input`]({{< relref "/docs/api/layers/core_layers/input#input-function" >}}) objects, but with the tensors that originate
from [`keras.Input`]({{< relref "/docs/api/layers/core_layers/input#input-function" >}}) objects. Under the hood, the layers and weights will
be shared across these models, so that user can train the `full_model`, and
use `backbone` or `activations` to do feature extraction.
The inputs and outputs of the model can be nested structures of tensors as
well, and the created models are standard Functional API models that support
all the existing APIs.

## By subclassing the `Model` class

In that case, you should define your
layers in `__init__()` and you should implement the model's forward pass
in `call()`.

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
model = MyModel()
```

If you subclass `Model`, you can optionally have
a `training` argument (boolean) in `call()`, which you can use to specify
a different behavior in training and inference:

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")
        self.dropout = keras.layers.Dropout(0.5)
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)
model = MyModel()
```

Once the model is created, you can config the model with losses and metrics
with `model.compile()`, train the model with `model.fit()`, or use the model
to do prediction with `model.predict()`.

## With the `Sequential` class

In addition, [`keras.Sequential`]({{< relref "/docs/api/models/sequential#sequential-class" >}}) is a special case of model where
the model is purely a stack of single-input, single-output layers.

```python
model = keras.Sequential([
    keras.Input(shape=(None, None, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=3),
])
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_processor.py#L39" >}}

### `CLIPProcessor` class

```python
keras_cv.models.feature_extractor.CLIPProcessor(vocabulary, merges, **kwargs)
```

CLIPProcessor is a utility class that provides functionality for processing
texts in the context of the CLIP (Contrastive Language-Image
Pretraining) model.

**Arguments**

- **input_resolution (int)**: The resolution of input images.
- **vocabulary (str)**: string or dict, maps token to integer ids. If it is a
  string, it should be the file path to a json file.
- **merges**: string or list, contains the merge rule. If it is a string, it
  should be the file path to merge rules. The merge rule file should
  have one merge rule per line.

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_text_model.py#L20" >}}

### `CLIPTextEncoder` class

```python
keras_cv.models.feature_extractor.CLIPTextEncoder(
    transformer_width,
    transformer_layers,
    transformer_heads,
    vocab_size,
    embed_dim,
    context_length,
    **kwargs
)
```

A model grouping layers into an object with training/inference features.

There are three ways to instantiate a `Model`:

## With the "Functional API"

You start from `Input`,
you chain layer calls to specify the model's forward pass,
and finally, you create your model from inputs and outputs:

```python
inputs = keras.Input(shape=(37,))
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

Note: Only dicts, lists, and tuples of input tensors are supported. Nested
inputs are not supported (e.g. lists of list or dicts of dict).

A new Functional API model can also be created by using the
intermediate tensors. This enables you to quickly extract sub-components
of the model.

**Example**

```python
inputs = keras.Input(shape=(None, None, 3))
processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
pooling = keras.layers.GlobalAveragePooling2D()(conv)
feature = keras.layers.Dense(10)(pooling)
full_model = keras.Model(inputs, feature)
backbone = keras.Model(processed, conv)
activations = keras.Model(conv, feature)
```

Note that the `backbone` and `activations` models are not
created with [`keras.Input`]({{< relref "/docs/api/layers/core_layers/input#input-function" >}}) objects, but with the tensors that originate
from [`keras.Input`]({{< relref "/docs/api/layers/core_layers/input#input-function" >}}) objects. Under the hood, the layers and weights will
be shared across these models, so that user can train the `full_model`, and
use `backbone` or `activations` to do feature extraction.
The inputs and outputs of the model can be nested structures of tensors as
well, and the created models are standard Functional API models that support
all the existing APIs.

## By subclassing the `Model` class

In that case, you should define your
layers in `__init__()` and you should implement the model's forward pass
in `call()`.

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
model = MyModel()
```

If you subclass `Model`, you can optionally have
a `training` argument (boolean) in `call()`, which you can use to specify
a different behavior in training and inference:

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")
        self.dropout = keras.layers.Dropout(0.5)
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)
model = MyModel()
```

Once the model is created, you can config the model with losses and metrics
with `model.compile()`, train the model with `model.fit()`, or use the model
to do prediction with `model.predict()`.

## With the `Sequential` class

In addition, [`keras.Sequential`]({{< relref "/docs/api/models/sequential#sequential-class" >}}) is a special case of model where
the model is purely a stack of single-input, single-output layers.

```python
model = keras.Sequential([
    keras.Input(shape=(None, None, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=3),
])
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_encoder.py#L19" >}}

### `QuickGELU` class

```python
keras_cv.models.feature_extractor.QuickGELU(**kwargs)
```

This is the class from which all layers inherit.

A layer is a callable object that takes as input one or more tensors and
that outputs one or more tensors. It involves _computation_, defined
in the `call()` method, and a _state_ (weight variables). State can be
created:

- in `__init__()`, for instance via `self.add_weight()`;
- in the optional `build()` method, which is invoked by the first
  `__call__()` to the layer, and supplies the shape(s) of the input(s),
  which may not have been known at initialization time.

Layers are recursively composable: If you assign a Layer instance as an
attribute of another Layer, the outer layer will start tracking the weights
created by the inner layer. Nested layers should be instantiated in the
`__init__()` method or `build()` method.

Users will just instantiate a layer and then treat it as a callable.

**Arguments**

- **trainable**: Boolean, whether the layer's variables should be trainable.
- **name**: String name of the layer.
- **dtype**: The dtype of the layer's computations and weights. Can also be a
  `keras.DTypePolicy`,
  which allows the computation and
  weight dtype to differ. Defaults to `None`. `None` means to use
  `keras.config.dtype_policy()`,
  which is a `float32` policy unless set to different value
  (via `keras.config.set_dtype_policy()`).

**Attributes**

- **name**: The name of the layer (string).
- **dtype**: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
- **variable_dtype**: Dtype of the layer's weights.
- **compute_dtype**: The dtype of the layer's computations.
  Layers automatically cast inputs to this dtype, which causes
  the computations and output to also be in this dtype.
  When mixed precision is used with a
  `keras.DTypePolicy`, this will be different
  than `variable_dtype`.
- **trainable_weights**: List of variables to be included in backprop.
- **non_trainable_weights**: List of variables that should not be
  included in backprop.
- **weights**: The concatenation of the lists trainable_weights and
  non_trainable_weights (in this order).
- **trainable**: Whether the layer should be trained (boolean), i.e.
  whether its potentially-trainable weights should be returned
  as part of `layer.trainable_weights`.
- **input_spec**: Optional (list of) `InputSpec` object(s) specifying the
  constraints on inputs that can be accepted by the layer.

We recommend that descendants of `Layer` implement the following methods:

- `__init__()`: Defines custom layer attributes, and creates layer weights
  that do not depend on input shapes, using `add_weight()`,
  or other state.
- `build(self, input_shape)`: This method can be used to create weights that
  depend on the shape(s) of the input(s), using `add_weight()`, or other
  state. `__call__()` will automatically build the layer
  (if it has not been built yet) by calling `build()`.
- `call(self, *args, **kwargs)`: Called in `__call__` after making
  sure `build()` has been called. `call()` performs the logic of applying
  the layer to the input arguments.
  Two reserved keyword arguments you can optionally use in `call()` are:
  1. `training` (boolean, whether the call is in inference mode or
     training mode).
  2. `mask` (boolean tensor encoding masked timesteps in the input,
     used e.g. in RNN layers).
     A typical signature for this method is `call(self, inputs)`, and user
     could optionally add `training` and `mask` if the layer need them.
- `get_config(self)`: Returns a dictionary containing the configuration
  used to initialize this layer. If the keys differ from the arguments
  in `__init__()`, then override `from_config(self)` as well.
  This method is used when saving
  the layer or a model that contains this layer.

**Examples**

Here's a basic example: a layer with two variables, `w` and `b`,
that returns `y = w . x + b`.
It shows how to implement `build()` and `call()`.
Variables set as attributes of a layer are tracked as weights
of the layers (in `layer.weights`).

```python
class SimpleDense(Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units
    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
    # Defines the computation
    def call(self, inputs):
        return ops.matmul(inputs, self.kernel) + self.bias
# Instantiates the layer.
linear_layer = SimpleDense(4)
# This will also call `build(input_shape)` and create the weights.
y = linear_layer(ops.ones((2, 2)))
assert len(linear_layer.weights) == 2
# These weights are trainable, so they're listed in `trainable_weights`:
assert len(linear_layer.trainable_weights) == 2
```

Besides trainable weights, updated via backpropagation during training,
layers can also have non-trainable weights. These weights are meant to
be updated manually during `call()`. Here's a example layer that computes
the running sum of its inputs:

```python
class ComputeSum(Layer):
  def __init__(self, input_dim):
      super(ComputeSum, self).__init__()
      # Create a non-trainable weight.
      self.total = self.add_weight(
        shape=(),
        initializer="zeros",
        trainable=False,
        name="total",
      )
  def call(self, inputs):
      self.total.assign(self.total + ops.sum(inputs))
      return self.total
my_sum = ComputeSum(2)
x = ops.ones((2, 2))
y = my_sum(x)
assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/feature_extractor/clip/clip_encoder.py#L28" >}}

### `ResidualAttention` class

```python
keras_cv.models.feature_extractor.ResidualAttention(
    proj_dim, num_heads, num_hidden_layers, **kwargs
)
```

This is the class from which all layers inherit.

A layer is a callable object that takes as input one or more tensors and
that outputs one or more tensors. It involves _computation_, defined
in the `call()` method, and a _state_ (weight variables). State can be
created:

- in `__init__()`, for instance via `self.add_weight()`;
- in the optional `build()` method, which is invoked by the first
  `__call__()` to the layer, and supplies the shape(s) of the input(s),
  which may not have been known at initialization time.

Layers are recursively composable: If you assign a Layer instance as an
attribute of another Layer, the outer layer will start tracking the weights
created by the inner layer. Nested layers should be instantiated in the
`__init__()` method or `build()` method.

Users will just instantiate a layer and then treat it as a callable.

**Arguments**

- **trainable**: Boolean, whether the layer's variables should be trainable.
- **name**: String name of the layer.
- **dtype**: The dtype of the layer's computations and weights. Can also be a
  `keras.DTypePolicy`,
  which allows the computation and
  weight dtype to differ. Defaults to `None`. `None` means to use
  `keras.config.dtype_policy()`,
  which is a `float32` policy unless set to different value
  (via `keras.config.set_dtype_policy()`).

**Attributes**

- **name**: The name of the layer (string).
- **dtype**: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
- **variable_dtype**: Dtype of the layer's weights.
- **compute_dtype**: The dtype of the layer's computations.
  Layers automatically cast inputs to this dtype, which causes
  the computations and output to also be in this dtype.
  When mixed precision is used with a
  `keras.DTypePolicy`, this will be different
  than `variable_dtype`.
- **trainable_weights**: List of variables to be included in backprop.
- **non_trainable_weights**: List of variables that should not be
  included in backprop.
- **weights**: The concatenation of the lists trainable_weights and
  non_trainable_weights (in this order).
- **trainable**: Whether the layer should be trained (boolean), i.e.
  whether its potentially-trainable weights should be returned
  as part of `layer.trainable_weights`.
- **input_spec**: Optional (list of) `InputSpec` object(s) specifying the
  constraints on inputs that can be accepted by the layer.

We recommend that descendants of `Layer` implement the following methods:

- `__init__()`: Defines custom layer attributes, and creates layer weights
  that do not depend on input shapes, using `add_weight()`,
  or other state.
- `build(self, input_shape)`: This method can be used to create weights that
  depend on the shape(s) of the input(s), using `add_weight()`, or other
  state. `__call__()` will automatically build the layer
  (if it has not been built yet) by calling `build()`.
- `call(self, *args, **kwargs)`: Called in `__call__` after making
  sure `build()` has been called. `call()` performs the logic of applying
  the layer to the input arguments.
  Two reserved keyword arguments you can optionally use in `call()` are:
  1. `training` (boolean, whether the call is in inference mode or
     training mode).
  2. `mask` (boolean tensor encoding masked timesteps in the input,
     used e.g. in RNN layers).
     A typical signature for this method is `call(self, inputs)`, and user
     could optionally add `training` and `mask` if the layer need them.
- `get_config(self)`: Returns a dictionary containing the configuration
  used to initialize this layer. If the keys differ from the arguments
  in `__init__()`, then override `from_config(self)` as well.
  This method is used when saving
  the layer or a model that contains this layer.

**Examples**

Here's a basic example: a layer with two variables, `w` and `b`,
that returns `y = w . x + b`.
It shows how to implement `build()` and `call()`.
Variables set as attributes of a layer are tracked as weights
of the layers (in `layer.weights`).

```python
class SimpleDense(Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units
    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
    # Defines the computation
    def call(self, inputs):
        return ops.matmul(inputs, self.kernel) + self.bias
# Instantiates the layer.
linear_layer = SimpleDense(4)
# This will also call `build(input_shape)` and create the weights.
y = linear_layer(ops.ones((2, 2)))
assert len(linear_layer.weights) == 2
# These weights are trainable, so they're listed in `trainable_weights`:
assert len(linear_layer.trainable_weights) == 2
```

Besides trainable weights, updated via backpropagation during training,
layers can also have non-trainable weights. These weights are meant to
be updated manually during `call()`. Here's a example layer that computes
the running sum of its inputs:

```python
class ComputeSum(Layer):
  def __init__(self, input_dim):
      super(ComputeSum, self).__init__()
      # Create a non-trainable weight.
      self.total = self.add_weight(
        shape=(),
        initializer="zeros",
        trainable=False,
        name="total",
      )
  def call(self, inputs):
      self.total.assign(self.total + ops.sum(inputs))
      return self.total
my_sum = ComputeSum(2)
x = ops.ones((2, 2))
y = my_sum(x)
assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```
