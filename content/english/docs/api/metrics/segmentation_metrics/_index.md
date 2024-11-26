---
title: Image segmentation metrics
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/iou_metrics.py#L145" >}}

### `IoU` class

```python
keras.metrics.IoU(
    num_classes,
    target_class_ids,
    name=None,
    dtype=None,
    ignore_class=None,
    sparse_y_true=True,
    sparse_y_pred=True,
    axis=-1,
)
```

Computes the Intersection-Over-Union metric for specific target classes.

Formula:

```python
iou = true_positives / (true_positives + false_positives + false_negatives)
```

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

Note, this class first computes IoUs for all individual classes, then
returns the mean of IoUs for the classes that are specified by
`target_class_ids`. If `target_class_ids` has only one id value, the IoU of
that specific class is returned.

**Arguments**

- **num_classes**: The possible number of labels the prediction task can have.
- **target_class_ids**: A tuple or list of target class ids for which the
  metric is returned. To compute IoU for a specific class, a list
  (or tuple) of a single id value should be provided.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.
- **ignore_class**: Optional integer. The ID of a class to be ignored during
  metric computation. This is useful, for example, in segmentation
  problems featuring a "void" class (commonly -1 or 255) in
  segmentation maps. By default (`ignore_class=None`), all classes are
  considered.
- **sparse_y_true**: Whether labels are encoded using integers or
  dense floating point vectors. If `False`, the `argmax` function
  is used to determine each sample's most likely associated label.
- **sparse_y_pred**: Whether predictions are encoded using integers or
  dense floating point vectors. If `False`, the `argmax` function
  is used to determine each sample's most likely associated label.
- **axis**: (Optional) -1 is the dimension containing the logits.
  Defaults to `-1`.

**Examples**

```console
>>> # cm = [[1, 1],
>>> #        [1, 1]]
>>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
>>> # iou = true_positives / (sum_row + sum_col - true_positives))
>>> # iou = [0.33, 0.33]
>>> m = keras.metrics.IoU(num_classes=2, target_class_ids=[0])
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
>>> m.result()
0.33333334
```

```console
>>> m.reset_state()
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
...                sample_weight=[0.3, 0.3, 0.3, 0.1])
>>> # cm = [[0.3, 0.3],
>>> #        [0.3, 0.1]]
>>> # sum_row = [0.6, 0.4], sum_col = [0.6, 0.4],
>>> # true_positives = [0.3, 0.1]
>>> # iou = [0.33, 0.14]
>>> m.result()
0.33333334
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.IoU(num_classes=2, target_class_ids=[0])])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/iou_metrics.py#L300" >}}

### `BinaryIoU` class

```python
keras.metrics.BinaryIoU(
    target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None
)
```

Computes the Intersection-Over-Union metric for class 0 and/or 1.

Formula:

```python
iou = true_positives / (true_positives + false_positives + false_negatives)
```

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

This class can be used to compute IoUs for a binary classification task
where the predictions are provided as logits. First a `threshold` is applied
to the predicted values such that those that are below the `threshold` are
converted to class 0 and those that are above the `threshold` are converted
to class 1.

IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
that are specified by `target_class_ids` is returned.

Note: with `threshold=0`, this metric has the same behavior as `IoU`.

**Arguments**

- **target_class_ids**: A tuple or list of target class ids for which the
  metric is returned. Options are `[0]`, `[1]`, or `[0, 1]`. With
  `[0]` (or `[1]`), the IoU metric for class 0 (or class 1,
  respectively) is returned. With `[0, 1]`, the mean of IoUs for the
  two classes is returned.
- **threshold**: A threshold that applies to the prediction logits to convert
  them to either predicted class 0 if the logit is below `threshold`
  or predicted class 1 if the logit is above `threshold`.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

**Example**

```console
>>> m = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
>>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7])
>>> m.result()
0.33333334
```

```console
>>> m.reset_state()
>>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7],
...                sample_weight=[0.2, 0.3, 0.4, 0.1])
>>> # cm = [[0.2, 0.4],
>>> #        [0.3, 0.1]]
>>> # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5],
>>> # true_positives = [0.2, 0.1]
>>> # iou = [0.222, 0.125]
>>> m.result()
0.17361112
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.BinaryIoU(
        target_class_ids=[0],
        threshold=0.5
    )]
)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/iou_metrics.py#L526" >}}

### `OneHotIoU` class

```python
keras.metrics.OneHotIoU(
    num_classes,
    target_class_ids,
    name=None,
    dtype=None,
    ignore_class=None,
    sparse_y_pred=False,
    axis=-1,
)
```

Computes the Intersection-Over-Union metric for one-hot encoded labels.

Formula:

```python
iou = true_positives / (true_positives + false_positives + false_negatives)
```

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

This class can be used to compute IoU for multi-class classification tasks
where the labels are one-hot encoded (the last axis should have one
dimension per class). Note that the predictions should also have the same
shape. To compute the IoU, first the labels and predictions are converted
back into integer format by taking the argmax over the class axis. Then the
same computation steps as for the base `IoU` class apply.

Note, if there is only one channel in the labels and predictions, this class
is the same as class `IoU`. In this case, use `IoU` instead.

Also, make sure that `num_classes` is equal to the number of classes in the
data, to avoid a "labels out of bound" error when the confusion matrix is
computed.

**Arguments**

- **num_classes**: The possible number of labels the prediction task can have.
- **target_class_ids**: A tuple or list of target class ids for which the
  metric is returned. To compute IoU for a specific class, a list
  (or tuple) of a single id value should be provided.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.
- **ignore_class**: Optional integer. The ID of a class to be ignored during
  metric computation. This is useful, for example, in segmentation
  problems featuring a "void" class (commonly -1 or 255) in
  segmentation maps. By default (`ignore_class=None`), all classes are
  considered.
- **sparse_y_pred**: Whether predictions are encoded using integers or
  dense floating point vectors. If `False`, the `argmax` function
  is used to determine each sample's most likely associated label.
- **axis**: (Optional) The dimension containing the logits. Defaults to `-1`.

**Example**

**Example**

```console
>>> y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
>>> y_pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
...                       [0.1, 0.4, 0.5]])
>>> sample_weight = [0.1, 0.2, 0.3, 0.4]
>>> m = keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2])
>>> m.update_state(
...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
>>> # cm = [[0, 0, 0.2+0.4],
>>> #       [0.3, 0, 0],
>>> #       [0, 0, 0.1]]
>>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
>>> # true_positives = [0, 0, 0.1]
>>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
>>> # mean_iou = (0 / (0.3 + 0.6 - 0) + 0.1 / (0.7 + 0.1 - 0.1)) / 2
>>> m.result()
0.071
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.OneHotIoU(
        num_classes=3,
        target_class_id=[1]
    )]
)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/iou_metrics.py#L643" >}}

### `OneHotMeanIoU` class

```python
keras.metrics.OneHotMeanIoU(
    num_classes, name=None, dtype=None, ignore_class=None, sparse_y_pred=False, axis=-1
)
```

Computes mean Intersection-Over-Union metric for one-hot encoded labels.

Formula:

```python
iou = true_positives / (true_positives + false_positives + false_negatives)
```

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

This class can be used to compute the mean IoU for multi-class
classification tasks where the labels are one-hot encoded (the last axis
should have one dimension per class). Note that the predictions should also
have the same shape. To compute the mean IoU, first the labels and
predictions are converted back into integer format by taking the argmax over
the class axis. Then the same computation steps as for the base `MeanIoU`
class apply.

Note, if there is only one channel in the labels and predictions, this class
is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.

Also, make sure that `num_classes` is equal to the number of classes in the
data, to avoid a "labels out of bound" error when the confusion matrix is
computed.

**Arguments**

- **num_classes**: The possible number of labels the prediction task can have.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.
- **ignore_class**: Optional integer. The ID of a class to be ignored during
  metric computation. This is useful, for example, in segmentation
  problems featuring a "void" class (commonly -1 or 255) in
  segmentation maps. By default (`ignore_class=None`), all classes are
  considered.
- **sparse_y_pred**: Whether predictions are encoded using natural numbers or
  probability distribution vectors. If `False`, the `argmax`
  function will be used to determine each sample's most likely
  associated label.
- **axis**: (Optional) The dimension containing the logits. Defaults to `-1`.

**Example**

**Example**

```console
>>> y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
>>> y_pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
...                       [0.1, 0.4, 0.5]])
>>> sample_weight = [0.1, 0.2, 0.3, 0.4]
>>> m = keras.metrics.OneHotMeanIoU(num_classes=3)
>>> m.update_state(
...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
>>> # cm = [[0, 0, 0.2+0.4],
>>> #       [0.3, 0, 0],
>>> #       [0, 0, 0.1]]
>>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
>>> # true_positives = [0, 0, 0.1]
>>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
>>> # mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
>>> m.result()
0.048
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.OneHotMeanIoU(num_classes=3)])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/iou_metrics.py#L422" >}}

### `MeanIoU` class

```python
keras.metrics.MeanIoU(
    num_classes,
    name=None,
    dtype=None,
    ignore_class=None,
    sparse_y_true=True,
    sparse_y_pred=True,
    axis=-1,
)
```

Computes the mean Intersection-Over-Union metric.

Formula:

```python
iou = true_positives / (true_positives + false_positives + false_negatives)
```

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

Note that this class first computes IoUs for all individual classes, then
returns the mean of these values.

**Arguments**

- **num_classes**: The possible number of labels the prediction task can have.
  This value must be provided, since a confusion matrix of dimension =
  [num\_classes, num\_classes] will be allocated.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.
- **ignore_class**: Optional integer. The ID of a class to be ignored during
  metric computation. This is useful, for example, in segmentation
  problems featuring a "void" class (commonly -1 or 255) in
  segmentation maps. By default (`ignore_class=None`), all classes are
  considered.
- **sparse_y_true**: Whether labels are encoded using integers or
  dense floating point vectors. If `False`, the `argmax` function
  is used to determine each sample's most likely associated label.
- **sparse_y_pred**: Whether predictions are encoded using integers or
  dense floating point vectors. If `False`, the `argmax` function
  is used to determine each sample's most likely associated label.
- **axis**: (Optional) The dimension containing the logits. Defaults to `-1`.

**Example**

**Example**

```console
>>> # cm = [[1, 1],
>>> #        [1, 1]]
>>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
>>> # iou = true_positives / (sum_row + sum_col - true_positives))
>>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
>>> m = keras.metrics.MeanIoU(num_classes=2)
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
>>> m.result()
0.33333334
```

```console
>>> m.reset_state()
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
...                sample_weight=[0.3, 0.3, 0.3, 0.1])
>>> m.result().numpy()
0.23809525
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.MeanIoU(num_classes=2)])
```
