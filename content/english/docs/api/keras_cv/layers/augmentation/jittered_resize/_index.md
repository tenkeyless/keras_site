---
title: JitteredResize layer
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/jittered_resize.py#L32" >}}

### `JitteredResize` class

```python
keras_cv.layers.JitteredResize(
    target_size,
    scale_factor,
    crop_size=None,
    bounding_box_format=None,
    interpolation="bilinear",
    seed=None,
    **kwargs
)
```

JitteredResize implements resize with scale distortion.

JitteredResize takes a three-step approach to size-distortion based image
augmentation. This technique is specifically tuned for object detection
pipelines. The layer takes an input of images and bounding boxes, both of
which may be ragged. It outputs a dense image tensor, ready to feed to a
model for training. As such this layer will commonly be the final step in an
augmentation pipeline.

The augmentation process is as follows:

The image is first scaled according to a randomly sampled scale factor. The
width and height of the image are then resized according to the sampled
scale. This is done to introduce noise into the local scale of features in
the image. A subset of the image is then cropped randomly according to
`crop_size`. This crop is then padded to be `target_size`. Bounding boxes
are translated and scaled according to the random scaling and random
cropping.

**Arguments**

- **target_size**: A tuple representing the output size of images.
- **scale_factor**: A tuple of two floats or a `keras_cv.FactorSampler`. For
  each augmented image a value is sampled from the provided range.
  This factor is used to scale the input image.
  To replicate the results of the MaskRCNN paper pass `(0.8, 1.25)`.
- **crop_size**: (Optional) the size of the image to crop from the scaled
  image, defaults to `target_size` when not provided.
- **bounding_box_format**: The format of bounding boxes of input boxes.
  Refer to
  https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
  for more details on supported bounding box formats.
- **interpolation**: String, the interpolation method, defaults to
  `"bilinear"`. Supports `"bilinear"`, `"nearest"`, `"bicubic"`,
  `"area"`, `"lanczos3"`, `"lanczos5"`, `"gaussian"`,
  `"mitchellcubic"`.
- **seed**: (Optional) integer to use as the random seed.

**Example**

```python
train_ds = load_object_detection_dataset()
jittered_resize = layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.8, 1.25),
    bounding_box_format="xywh",
)
train_ds = train_ds.map(
    jittered_resize, num_parallel_calls=tf.data.AUTOTUNE
)
# images now are (640, 640, 3)
# an example using crop size
train_ds = load_object_detection_dataset()
jittered_resize = layers.JitteredResize(
    target_size=(640, 640),
    crop_size=(250, 250),
    scale_factor=(0.8, 1.25),
    bounding_box_format="xywh",
)
train_ds = train_ds.map(
    jittered_resize, num_parallel_calls=tf.data.AUTOTUNE
)
# images now are (640, 640, 3), but they were resized from a 250x250 crop.
```
