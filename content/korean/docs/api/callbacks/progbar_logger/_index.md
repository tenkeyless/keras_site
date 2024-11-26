---
title: ProgbarLogger
toc: true
weight: 12
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/callbacks/progbar_logger.py#L7" >}}

### `ProgbarLogger` class

```python
keras.callbacks.ProgbarLogger()
```

Callback that prints metrics to stdout.

**Arguments**

- **count_mode**: One of `"steps"` or `"samples"`.
  Whether the progress bar should
  count samples seen or steps (batches) seen.

**Raises**

- **ValueError**: In case of invalid `count_mode`.
