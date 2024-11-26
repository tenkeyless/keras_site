---
title: CIoU Loss
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/losses/ciou_loss.py#L22" >}}

### `CIoULoss` class

```python
keras_cv.losses.CIoULoss(bounding_box_format, eps=1e-07, **kwargs)
```

Implements the Complete IoU (CIoU) Loss

CIoU loss is an extension of GIoU loss, which further improves the IoU
optimization for object detection. CIoU loss not only penalizes the
bounding box coordinates but also considers the aspect ratio and center
distance of the boxes. The length of the last dimension should be 4 to
represent the bounding boxes.

**Arguments**

- **bounding_box_format**: a case-insensitive string (for example, "xyxy").
  Each bounding box is defined by these 4 values. For detailed
  information on the supported formats, see the [KerasCV bounding box
  documentation]({{< relref "/docs/api/keras_cv/bounding_box/formats/" >}}).
- **eps**: A small value added to avoid division by zero and stabilize
  calculations.

**References**

- [CIoU paper](https://arxiv.org/pdf/2005.03572.pdf)

**Example**

```python
y_true = np.random.uniform(
    size=(5, 10, 5),
    low=0,
    high=10)
y_pred = np.random.uniform(
    (5, 10, 4),
    low=0,
    high=10)
loss = keras_cv.losses.CIoULoss()
loss(y_true, y_pred).numpy()
```

Usage with the `compile()` API:

```python
model.compile(optimizer='adam', loss=CIoULoss())
```
