---
title: Convert a bounding box dictionary to -1 padded Dense tensors
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/to_dense.py#L40" >}}

### `to_dense` function

```python
keras_cv.bounding_box.to_dense(bounding_boxes, max_boxes=None, default_value=-1)
```

to_dense converts bounding boxes to Dense tensors

**Arguments**

- **bounding_boxes**: bounding boxes in KerasCV dictionary format.
- **max_boxes**: the maximum number of boxes, used to pad tensors to a given
  shape. This can be used to make object detection pipelines TPU
  compatible.
- **default_value**: the default value to pad bounding boxes with. defaults
  to -1.
