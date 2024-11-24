---
title: resizing
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/resizing.py#L37" >}}

### `Resizing` class

`keras_cv.layers.Resizing(     height,     width,     interpolation="bilinear",     crop_to_aspect_ratio=False,     pad_to_aspect_ratio=False,     bounding_box_format=None,     **kwargs )`

A preprocessing layer which resizes images.

This layer resizes an image input to a target height and width. The input should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"` format. Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype. By default, the layer will output floats.

This layer can be called on tf.RaggedTensor batches of input images of distinct sizes, and will resize the outputs to dense tensors of uniform size.

For an overview and full list of preprocessing layers, see the preprocessing [guide](https://keras.io/guides/preprocessing_layers).

**Arguments**

- **height**: Integer, the height of the output shape.
- **width**: Integer, the width of the output shape.
- **interpolation**: String, the interpolation method, defaults to `"bilinear"`. Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
- **crop_to_aspect_ratio**: If True, resize the images without aspect ratio distortion. When the original aspect ratio differs from the target aspect ratio, the output image will be cropped to return the largest possible window in the image (of size `(height, width)`) that matches the target aspect ratio. By default, (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
- **pad_to_aspect_ratio**: If True, resize the images without aspect ratio distortion. When the original aspect ratio differs from the target aspect ratio, the output image will be padded to return the largest possible resize of the image (of size `(height, width)`) that matches the target aspect ratio. By default, (`pad_to_aspect_ratio=False`), aspect ratio may not be preserved.
- **bounding_box_format**: The format of bounding boxes of input dataset. Refer to https://github.com/keras-team/keras-cv/blob/master/keras\_cv/bounding\_box/converters.py for more details on supported bounding box formats.

---
