---
title: Convert a bounding box dictionary batched Ragged tensors
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/to_ragged.py#L22" >}}

### `to_ragged` function

```python
keras_cv.bounding_box.to_ragged(bounding_boxes, sentinel=-1, dtype=tf.float32)
```

converts a Dense padded bounding box [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) to a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor).

Bounding boxes are ragged tensors in most use cases. Converting them to a
dense tensor makes it easier to work with Tensorflow ecosystem.
This function can be used to filter out the masked out bounding boxes by
checking for padded sentinel value of the class_id axis of the
bounding_boxes.

**Example**

```python
bounding_boxes = {
    "boxes": tf.constant([[2, 3, 4, 5], [0, 1, 2, 3]]),
    "classes": tf.constant([[-1, 1]]),
}
bounding_boxes = bounding_box.to_ragged(bounding_boxes)
print(bounding_boxes)
# {
#     "boxes": [[0, 1, 2, 3]],
#     "classes": [[1]]
# }
```

**Arguments**

- **bounding_boxes**: a Tensor of bounding boxes. May be batched, or
  unbatched.
- **sentinel**: The value indicating that a bounding box does not exist at the
  current index, and the corresponding box is padding, defaults to -1.
- **dtype**: the data type to use for the underlying Tensors.

**Returns**

dictionary of [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) or 'tf.Tensor' containing the filtered
bounding boxes.
