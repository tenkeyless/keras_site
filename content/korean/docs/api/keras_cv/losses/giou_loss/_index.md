---
title: GIoU Loss
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/losses/giou_loss.py#L23" >}}

### `GIoULoss` class

```python
keras_cv.losses.GIoULoss(bounding_box_format, axis=-1, **kwargs)
```

Implements the Generalized IoU Loss

GIoU loss is a modified IoU loss commonly used for object detection. This
loss aims to directly optimize the IoU score between true boxes and
predicted boxes. GIoU loss adds a penalty term to the IoU loss that takes in
account the area of the smallest box enclosing both the boxes being
considered for the iou. The length of the last dimension should be 4 to
represent the bounding boxes.

**Arguments**

- **bounding_box_format**: a case-insensitive string (for example, "xyxy").
  Each bounding box is defined by these 4 values.For detailed
  information on the supported formats, see the [KerasCV bounding box
  documentation]({{< relref "/docs/api/keras_cv/bounding_box/formats/" >}}).
- **axis**: the axis along which to mean the ious, defaults to -1.

**References**

- [GIoU paper](https://arxiv.org/pdf/1902.09630)
  - [TFAddons Implementation](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/GIoULoss)

**Example**

```python
y_true = np.random.uniform(size=(5, 10, 5), low=0, high=10)
y_pred = np.random.uniform(size=(5, 10, 4), low=0, high=10)
loss = GIoULoss(bounding_box_format = "xyWH")
loss(y_true, y_pred).numpy()
```

Usage with the `compile()` API:

```python
model.compile(optimizer='adam', loss=keras_cv.losses.GIoULoss())
```
