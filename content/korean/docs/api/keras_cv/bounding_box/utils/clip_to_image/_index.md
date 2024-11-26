---
title: Clip bounding boxes to be within the bounds of provided images
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/utils.py#L67" >}}

### `clip_to_image` function

```python
keras_cv.bounding_box.clip_to_image(
    bounding_boxes, bounding_box_format, images=None, image_shape=None
)
```

clips bounding boxes to image boundaries.

`clip_to_image()` clips bounding boxes that have coordinates out of bounds
of an image down to the boundaries of the image. This is done by converting
the bounding box to relative formats, then clipping them to the `[0, 1]`
range. Additionally, bounding boxes that end up with a zero area have their
class ID set to -1, indicating that there is no object present in them.

**Arguments**

- **bounding_boxes**: bounding box tensor to clip.
- **bounding_box_format**: the KerasCV bounding box format the bounding boxes
  are in.
- **images**: list of images to clip the bounding boxes to.
- **image_shape**: the shape of the images to clip the bounding boxes to.
