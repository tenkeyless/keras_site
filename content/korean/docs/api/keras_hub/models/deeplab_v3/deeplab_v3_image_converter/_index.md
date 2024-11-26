---
title: DeepLabV3ImageConverter
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/models/deeplab_v3/deeplab_v3_image_converter.py#L8" >}}

### `DeepLabV3ImageConverter` class

```python
keras_hub.layers.DeepLabV3ImageConverter(
    image_size=None,
    scale=None,
    offset=None,
    crop_to_aspect_ratio=True,
    interpolation="bilinear",
    data_format=None,
    **kwargs
)
```

Preprocess raw images into model ready inputs.

This class converts from raw images to model ready inputs. This conversion
proceeds in the following steps:

1. Resize the image using to `image_size`. If `image_size` is `None`, this
   step will be skipped.
2. Rescale the image by multiplying by `scale`, which can be either global
   or per channel. If `scale` is `None`, this step will be skipped.
3. Offset the image by adding `offset`, which can be either global
   or per channel. If `offset` is `None`, this step will be skipped.

The layer will take as input a raw image tensor in the channels last or
channels first format, and output a preprocessed image input for modeling.
This tensor can be batched (rank 4), or unbatched (rank 3).

This layer can be used with the `from_preset()` constructor to load a layer
that will rescale and resize an image for a specific pretrained model.
Using the layer this way allows writing preprocessing code that does not
need updating when switching between model checkpoints.

**Arguments**

- **image_size**: `(int, int)` tuple or `None`. The output size of the image,
  not including the channels axis. If `None`, the input will not be
  resized.
- **scale**: float, tuple of floats, or `None`. The scale to apply to the
  inputs. If `scale` is a single float, the entire input will be
  multiplied by `scale`. If `scale` is a tuple, it's assumed to
  contain per-channel scale value multiplied against each channel of
  the input images. If `scale` is `None`, no scaling is applied.
- **offset**: float, tuple of floats, or `None`. The offset to apply to the
  inputs. If `offset` is a single float, the entire input will be
  summed with `offset`. If `offset` is a tuple, it's assumed to
  contain per-channel offset value summed against each channel of the
  input images. If `offset` is `None`, no scaling is applied.
- **crop_to_aspect_ratio**: If `True`, resize the images without aspect
  ratio distortion. When the original aspect ratio differs
  from the target aspect ratio, the output image will be
  cropped so as to return the
  largest possible window in the image (of size `(height, width)`)
  that matches the target aspect ratio. By default
  (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
- **interpolation**: String, the interpolation method.
  Supports `"bilinear"`, `"nearest"`, `"bicubic"`,
  `"lanczos3"`, `"lanczos5"`. Defaults to `"bilinear"`.
- **data_format**: String, either `"channels_last"` or `"channels_first"`.
  The ordering of the dimensions in the inputs. `"channels_last"`
  corresponds to inputs with shape `(batch, height, width, channels)`
  while `"channels_first"` corresponds to inputs with shape
  `(batch, channels, height, width)`. It defaults to the
  `image_data_format` value found in your Keras config file at
  `~/.keras/keras.json`. If you never set it, then it will be
  `"channels_last"`.

**Examples**

```python
# Resize raw images and scale them to [0, 1].
converter = keras_hub.layers.ImageConverter(
    image_size=(128, 128),
    scale=1. / 255,
)
converter(np.random.randint(0, 256, size=(2, 512, 512, 3)))
# Resize images to the specific size needed for a PaliGemma preset.
converter = keras_hub.layers.ImageConverter.from_preset(
    "pali_gemma_3b_224"
)
converter(np.random.randint(0, 256, size=(2, 512, 512, 3)))
```

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/preprocessing/image_converter.py#L189" >}}

### `from_preset` method

```python
DeepLabV3ImageConverter.from_preset(preset, **kwargs)
```

Instantiate a [`keras_hub.layers.ImageConverter`]({{< relref "/docs/api/keras_hub/preprocessing_layers/image_converter#imageconverter-class" >}}) from a model preset.

A preset is a directory of configs, weights and other file assets used
to save and load a pre-trained model. The `preset` can be passed as
one of:

1. a built-in preset identifier like `'pali_gemma_3b_224'`
2. a Kaggle Models handle like
   `'kaggle://user/paligemma/keras/pali_gemma_3b_224'`
3. a Hugging Face handle like `'hf://user/pali_gemma_3b_224'`
4. a path to a local preset directory like `'./pali_gemma_3b_224'`

You can run `cls.presets.keys()` to list all built-in presets available
on the class.

**Arguments**

- **preset**: string. A built-in preset identifier, a Kaggle Models
  handle, a Hugging Face handle, or a path to a local directory.
- **load_weights**: bool. If `True`, the weights will be loaded into the
  model architecture. If `False`, the weights will be randomly
  initialized.

**Examples**

```python
batch = np.random.randint(0, 256, size=(2, 512, 512, 3))
# Resize images for `"pali_gemma_3b_224"`.
converter = keras_hub.layers.ImageConverter.from_preset(
    "pali_gemma_3b_224"
)
converter(batch) # # Output shape (2, 224, 224, 3)
# Resize images for `"pali_gemma_3b_448"` without cropping.
converter = keras_hub.layers.ImageConverter.from_preset(
    "pali_gemma_3b_448",
    crop_to_aspect_ratio=False,
)
converter(batch) # # Output shape (2, 448, 448, 3)
```

| Preset name                        | Parameters | Description                                                                                                                                                                                     |
| ---------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| deeplab_v3_plus_resnet50_pascalvoc | 39.19M     | DeepLabV3+ model with ResNet50 as image encoder and trained on augmented Pascal VOC dataset by Semantic Boundaries Dataset(SBD)which is having categorical accuracy of 90.01 and 0.63 Mean IoU. |
