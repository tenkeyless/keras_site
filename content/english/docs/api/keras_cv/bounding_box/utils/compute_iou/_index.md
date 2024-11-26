---
title: Compute intersection over union of bounding boxes
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/iou.py#L64" >}}

### `compute_iou` function

```python
keras_cv.bounding_box.compute_iou(
    boxes1,
    boxes2,
    bounding_box_format,
    use_masking=False,
    mask_val=-1,
    images=None,
    image_shape=None,
)
```

Computes a lookup table vector containing the ious for a given set boxes.

The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`] if
boxes are unbatched and by [`batch`, `boxes1_index`,`boxes2_index`] if the
boxes are batched.

The users can pass `boxes1` and `boxes2` to be different ranks. For example:

1. `boxes1`: [batch\_size, M, 4], `boxes2`: [batch\_size, N, 4] -> return
   [batch\_size, M, N].
2. `boxes1`: [batch\_size, M, 4], `boxes2`: [N, 4] -> return
   [batch\_size, M, N]
3. `boxes1`: [M, 4], `boxes2`: [batch\_size, N, 4] -> return
   [batch\_size, M, N]
4. `boxes1`: [M, 4], `boxes2`: [N, 4] -> return [M, N]

**Arguments**

- **boxes1**: a list of bounding boxes in 'corners' format. Can be batched or
  unbatched.
- **boxes2**: a list of bounding boxes in 'corners' format. Can be batched or
  unbatched.
- **bounding_box_format**: a case-insensitive string which is one of `"xyxy"`,
  `"rel_xyxy"`, `"xyWH"`, `"center_xyWH"`, `"yxyx"`, `"rel_yxyx"`.
  For detailed information on the supported format, see the
  [KerasCV bounding box documentation]({{< relref "/docs/api/keras_cv/bounding_box/formats/" >}}).
- **use_masking**: whether masking will be applied. This will mask all `boxes1`
  or `boxes2` that have values less than 0 in all its 4 dimensions.
  Default to `False`.
- **mask_val**: int to mask those returned IOUs if the masking is True, defaults
  to -1.

**Returns**

- **iou_lookup_table**: a vector containing the pairwise ious of boxes1 and
  boxes2.
