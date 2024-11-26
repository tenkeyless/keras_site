---
title: Ensure that your bounding boxes comply with the bounding box spec
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/validate_format.py#L19" >}}

### `validate_format` function

```python
keras_cv.bounding_box.validate_format(bounding_boxes, variable_name="bounding_boxes")
```

validates that a given set of bounding boxes complies with KerasCV
format.

For a set of bounding boxes to be valid it must satisfy the following
conditions:

- `bounding_boxes` must be a dictionary
- contains keys `"boxes"` and `"classes"`
- each entry must have matching first two dimensions; representing the batch
  axis and the number of boxes per image axis.
- either both `"boxes"` and `"classes"` are batched, or both are unbatched.

Additionally, one of the following must be satisfied:

- `"boxes"` and `"classes"` are both Ragged
- `"boxes"` and `"classes"` are both Dense
- `"boxes"` and `"classes"` are unbatched

**Arguments**

- **bounding_boxes**: dictionary of bounding boxes according to KerasCV
  format.

**Raises**

ValueError if any of the above conditions are not met
