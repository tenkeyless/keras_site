---
title: DeepLabV3ImageSegmenterPreprocessor layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/deeplab_v3/deeplab_v3_image_segmeter_preprocessor.py#L13" >}}

### `DeepLabV3ImageSegmenterPreprocessor` class

```python
keras_hub.models.DeepLabV3ImageSegmenterPreprocessor(
    image_converter=None, resize_output_mask=False, **kwargs
)
```

Base class for image segmentation preprocessing layers.

`ImageSegmenterPreprocessor` wraps a
[`keras_hub.layers.ImageConverter`]({{< relref "/docs/api/keras_hub/preprocessing_layers/image_converter#imageconverter-class" >}}) to create a preprocessing layer for
image segmentation tasks. It is intended to be paired with a
[`keras_hub.models.ImageSegmenter`]({{< relref "/docs/api/keras_hub/base_classes/image_segmenter#imagesegmenter-class" >}}) task.

All `ImageSegmenterPreprocessor` instances take three inputs: `x`, `y`, and
`sample_weight`.

- `x`: The first input, should always be included. It can be an image or
  a batch of images.
- `y`: (Optional) Usually the segmentation mask(s), if `resize_output_mask`
  is set to `True` this will be resized to input image shape else will be
  passed through unaltered.
- `sample_weight`: (Optional) Will be passed through unaltered.
- `resize_output_mask` bool: If set to `True` the output mask will be resized to the same size as the input image. Defaults to `False`.

The layer will output either `x`, an `(x, y)` tuple if labels were provided,
or an `(x, y, sample_weight)` tuple if labels and sample weight were
provided. `x` will be the input images after all model preprocessing has
been applied.

All `ImageSegmenterPreprocessor` tasks include a `from_preset()`
constructor which can be used to load a pre-trained config.
You can call the `from_preset()` constructor directly on this base class, in
which case the correct class for your model will be automatically
instantiated.

Examples.

```python
preprocessor = keras_hub.models.ImageSegmenterPreprocessor.from_preset(
    "deeplabv3_resnet50",
)
# Resize a single image for the model.
x = np.ones((512, 512, 3))
x = preprocessor(x)
# Resize an image and its mask.
x, y = np.ones((512, 512, 3)), np.zeros((512, 512, 1))
x, y = preprocessor(x, y)
# Resize a batch of images and masks.
x, y = [np.ones((512, 512, 3)), np.zeros((512, 512, 3))],
       [np.ones((512, 512, 1)), np.zeros((512, 512, 1))]
x, y = preprocessor(x, y)
# Use a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/preprocessor.py#L132" >}}

### `from_preset` method

```python
DeepLabV3ImageSegmenterPreprocessor.from_preset(
    preset, config_file="preprocessor.json", **kwargs
)
```

Instantiate a [`keras_hub.models.Preprocessor`]({{< relref "/docs/api/keras_hub/base_classes/preprocessor#preprocessor-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'bert_base_en'`
2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
3. a Hugging Face handle like `'hf://user/bert_base_en'`
4. a path to a local preset directory like `'./bert_base_en'`

For any `Preprocessor` subclass, you can run `cls.presets.keys()` to
list all built-in presets available on the class.

As there are usually multiple preprocessing classes for a given model,
this method should be called on a specific subclass like
`keras_hub.models.BertTextClassifierPreprocessor.from_preset()`.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.

**Examples**

```python
# Load a preprocessor for Gemma generation.
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en",
)
# Load a preprocessor for Bert classification.
preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en",
)
```

| Preset name                        | Parameters | Description                                                                                                                                                                                     |
| ---------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| deeplab_v3_plus_resnet50_pascalvoc | 39.19M     | DeepLabV3+ model with ResNet50 as image encoder and trained on augmented Pascal VOC dataset by Semantic Boundaries Dataset(SBD)which is having categorical accuracy of 90.01 and 0.63 Mean IoU. |

### `image_converter` property

```python
keras_hub.models.DeepLabV3ImageSegmenterPreprocessor.image_converter
```

The image converter used to preprocess image data.
