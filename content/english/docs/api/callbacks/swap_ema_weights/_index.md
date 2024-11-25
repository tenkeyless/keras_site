---
title: SwapEMAWeights
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/callbacks/swap_ema_weights.py#L7" >}}

### `SwapEMAWeights` class

```python
keras.callbacks.SwapEMAWeights(swap_on_epoch=False)
```

Swaps model weights and EMA weights before and after evaluation.

This callbacks replaces the model's weight values with the values of
the optimizer's EMA weights (the exponential moving average of the past
model weights values, implementing "Polyak averaging") before model
evaluation, and restores the previous weights after evaluation.

The `SwapEMAWeights` callback is to be used in conjunction with
an optimizer that sets `use_ema=True`.

Note that the weights are swapped in-place in order to save memory.
The behavior is undefined if you modify the EMA weights
or model weights in other callbacks.

**Example**

```python
optimizer = SGD(use_ema=True)
model.compile(optimizer=optimizer, loss=..., metrics=...)
model.fit(X_train, Y_train, callbacks=[SwapEMAWeights()])
model.fit(
    X_train,
    Y_train,
    callbacks=[SwapEMAWeights(swap_on_epoch=True), ModelCheckpoint(...)]
)
```

**Arguments**

- **swap_on_epoch**: whether to perform swapping at `on_epoch_begin()`
  and `on_epoch_end()`. This is useful if you want to use
  EMA weights for other callbacks such as `ModelCheckpoint`.
  Defaults to `False`.
