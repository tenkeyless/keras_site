---
title: DenseNetImageClassifier model
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/densenet/densenet_image_classifier.py#L9" >}}

### `DenseNetImageClassifier` class

```python
keras_hub.models.DenseNetImageClassifier(
    backbone,
    num_classes,
    preprocessor=None,
    pooling="avg",
    activation=None,
    dropout=0.0,
    head_dtype=None,
    **kwargs
)
```

Base class for all image classification tasks.

`ImageClassifier` tasks wrap a [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) and
a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) to create a model that can be used for
image classification. `ImageClassifier` tasks take an additional
`num_classes` argument, controlling the number of predicted output classes.

To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
labels where `x` is a string and `y` is a integer from `[0, num_classes)`.
All `ImageClassifier` tasks include a `from_preset()` constructor which can
be used to load a pre-trained config and weights.

**Arguments**

- **backbone**: A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) instance or a [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}).
- **num_classes**: int. The number of classes to predict.
- **preprocessor**: `None`, a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) instance,
  a `keras.Layer` instance, or a callable. If `None` no preprocessing
  will be applied to the inputs.
- **pooling**: `"avg"` or `"max"`. The type of pooling to apply on backbone
  output. Defaults to average pooling.
- **activation**: `None`, str, or callable. The activation function to use on
  the `Dense` layer. Set `activation=None` to return the output
  logits. Defaults to `"softmax"`.
- **head_dtype**: `None`, str, or `keras.mixed_precision.DTypePolicy`. The
  dtype to use for the classification head's computations and weights.

**Examples**

Call `predict()` to run inference.

```python
# Load preset and train
images = np.random.randint(0, 256, size=(2, 224, 224, 3))
classifier = keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet"
)
classifier.predict(images)
```

Call `fit()` on a single batch.

```python
# Load preset and train
images = np.random.randint(0, 256, size=(2, 224, 224, 3))
labels = [0, 3]
classifier = keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet"
)
classifier.fit(x=images, y=labels, batch_size=2)
```

Call `fit()` with custom loss, optimizer and backbone.

```python
classifier = keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet"
)
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
)
classifier.backbone.trainable = False
classifier.fit(x=images, y=labels, batch_size=2)
```

Custom backbone.

```python
images = np.random.randint(0, 256, size=(2, 224, 224, 3))
labels = [0, 3]
backbone = keras_hub.models.ResNetBackbone(
    stackwise_num_filters=[64, 64, 64],
    stackwise_num_blocks=[2, 2, 2],
    stackwise_num_strides=[1, 2, 2],
    block_type="basic_block",
    use_pre_activation=True,
    pooling="avg",
)
classifier = keras_hub.models.ImageClassifier(
    backbone=backbone,
    num_classes=4,
)
classifier.fit(x=images, y=labels, batch_size=2)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/task.py#L129" >}}

### `from_preset` method

```python
DenseNetImageClassifier.from_preset(preset, load_weights=True, **kwargs)
```

Instantiate a [`keras_hub.models.Task`]({{< relref "/docs/api/keras_hub/base_classes/task#task-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Task` subclass, you can run `cls.presets.keys()` to list all
built-in presets available on the class.

This constructor can be called in one of two ways. Either from a task
specific base class like `keras_hub.models.CausalLM.from_preset()`, or
from a model class like `keras_hub.models.BertTextClassifier.from_preset()`.
If calling from the a base class, the subclass of the returning object
will be inferred from the config in the preset directory.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, saved weights will be loaded into
  the model architecture. If `False`, all weights will be
  randomly initialized.

**Examples**

```python
# Load a Gemma generative task.
causal_lm = keras_hub.models.CausalLM.from_preset(
    "gemma_2b_en",
)
# Load a Bert classification task.
model = keras_hub.models.TextClassifier.from_preset(
    "bert_base_en",
    num_classes=2,
)
```

| Preset name           | Parameters | Description                                                                              |
| --------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| densenet_121_imagenet | 7.04M      | 121-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution. |
| densenet_169_imagenet | 12.64M     | 169-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution. |
| densenet_201_imagenet | 18.32M     | 201-layer DenseNet model pre-trained on the ImageNet 1k dataset at a 224x224 resolution. |

### `backbone` property

```python
keras_hub.models.DenseNetImageClassifier.backbone
```

A [`keras_hub.models.Backbone`]({{< relref "/docs/api/keras_hub/base_classes/backbone#backbone-class" >}}) model with the core architecture.

### `preprocessor` property

```python
keras_hub.models.DenseNetImageClassifier.preprocessor
```

A [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) layer used to preprocess input.
