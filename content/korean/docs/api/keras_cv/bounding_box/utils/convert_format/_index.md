---
title: Convert bounding box formats
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/formats.py#L63" >}}

### `CENTER_XYWH` class

```python
keras_cv.bounding_box.CENTER_XYWH()
```

CENTER_XYWH contains axis indices for the CENTER_XYWH format.

All values in the CENTER_XYWH format should be absolute pixel values.

The CENTER_XYWH format consists of the following required indices:

- X: X coordinate of the center of the bounding box
- Y: Y coordinate of the center of the bounding box
- WIDTH: width of the bounding box
- HEIGHT: height of the bounding box

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/formats.py#L83" >}}

### `XYWH` class

```python
keras_cv.bounding_box.XYWH()
```

XYWH contains axis indices for the XYWH format.

All values in the XYWH format should be absolute pixel values.

The XYWH format consists of the following required indices:

- X: X coordinate of the left of the bounding box
- Y: Y coordinate of the top of the bounding box
- WIDTH: width of the bounding box
- HEIGHT: height of the bounding box

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/formats.py#L103" >}}

### `REL_XYWH` class

```python
keras_cv.bounding_box.REL_XYWH()
```

REL_XYWH contains axis indices for the XYWH format.

REL_XYXY is like XYWH, but each value is relative to the width and height of
the origin image. Values are percentages of the origin images' width and
height respectively.

- X: X coordinate of the left of the bounding box
- Y: Y coordinate of the top of the bounding box
- WIDTH: width of the bounding box
- HEIGHT: height of the bounding box

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/formats.py#L21" >}}

### `XYXY` class

```python
keras_cv.bounding_box.XYXY()
```

XYXY contains axis indices for the XYXY format.

All values in the XYXY format should be absolute pixel values.

The XYXY format consists of the following required indices:

- LEFT: left of the bounding box
- TOP: top of the bounding box
- RIGHT: right of the bounding box
- BOTTOM: bottom of the bounding box

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/formats.py#L41" >}}

### `REL_XYXY` class

```python
keras_cv.bounding_box.REL_XYXY()
```

REL_XYXY contains axis indices for the REL_XYXY format.

REL_XYXY is like XYXY, but each value is relative to the width and height of
the origin image. Values are percentages of the origin images' width and
height respectively.

The REL_XYXY format consists of the following required indices:

- LEFT: left of the bounding box
- TOP: top of the bounding box
- RIGHT: right of the bounding box
- BOTTOM: bottom of the bounding box

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/formats.py#L123" >}}

### `YXYX` class

```python
keras_cv.bounding_box.YXYX()
```

YXYX contains axis indices for the YXYX format.

All values in the YXYX format should be absolute pixel values.

The YXYX format consists of the following required indices:

- TOP: top of the bounding box
- LEFT: left of the bounding box
- BOTTOM: bottom of the bounding box
- RIGHT: right of the bounding box

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/bounding_box/formats.py#L143" >}}

### `REL_YXYX` class

```python
keras_cv.bounding_box.REL_YXYX()
```

REL_YXYX contains axis indices for the REL_YXYX format.

REL_YXYX is like YXYX, but each value is relative to the width and height of
the origin image. Values are percentages of the origin images' width and
height respectively.

The REL_YXYX format consists of the following required indices:

- TOP: top of the bounding box
- LEFT: left of the bounding box
- BOTTOM: bottom of the bounding box
- RIGHT: right of the bounding box
