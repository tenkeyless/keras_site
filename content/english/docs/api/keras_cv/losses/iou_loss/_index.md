---
title: IoU Loss
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/losses/iou_loss.py#L24" >}}

### `IoULoss` class

```python
keras_cv.losses.IoULoss(bounding_box_format, mode="log", axis=-1, **kwargs)
```

Implements the IoU Loss

IoU loss is commonly used for object detection. This loss aims to directly
optimize the IoU score between true boxes and predicted boxes. The length of
the last dimension should be 4 to represent the bounding boxes. This loss
uses IoUs according to box pairs and therefore, the number of boxes in both
y_true and y_pred are expected to be equal i.e. the ith
y_true box in a batch will be compared the ith y_pred box.

**Arguments**

- **bounding_box_format**: a case-insensitive string (for example, "xyxy").
  Each bounding box is defined by these 4 values. For detailed
  information on the supported formats, see the
  [KerasCV bounding box documentation]({{< relref "/docs/api/keras_cv/bounding_box/formats/" >}}).
- **mode**: must be one of
  - `"linear"`. The loss will be calculated as 1 - iou
  - `"quadratic"`. The loss will be calculated as 1 - iou2
  - `"log"`. The loss will be calculated as -ln(iou)
    Defaults to "log".
- **axis**: the axis along which to mean the ious, defaults to -1.

**References**

- [UnitBox paper](https://arxiv.org/pdf/1608.01471)

**Example**

```python
y_true = np.random.uniform(size=(5, 10, 5), low=10, high=10)
y_pred = np.random.uniform(size=(5, 10, 5), low=10, high=10)
loss = IoULoss(bounding_box_format = "xyWH")
loss(y_true, y_pred)
```

Usage with the `compile()` API:

```python
model.compile(optimizer='adam', loss=keras_cv.losses.IoULoss())
```
