---
title: Callbacks API
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

A callback is an object that can perform actions at various stages of training
(e.g. at the start or end of an epoch, before or after a single batch, etc).

You can use callbacks to:

- Write TensorBoard logs after every batch of training to monitor your metrics
- Periodically save your model to disk
- Do early stopping
- Get a view on internal states and statistics of a model during training
- ...and more

## Available callbacks

- [Base Callback class]({{< relref "/docs/api/callbacks/base_callback/" >}})
- [ModelCheckpoint]({{< relref "/docs/api/callbacks/model_checkpoint/" >}})
- [BackupAndRestore]({{< relref "/docs/api/callbacks/backup_and_restore/" >}})
- [TensorBoard]({{< relref "/docs/api/callbacks/tensorboard/" >}})
- [EarlyStopping]({{< relref "/docs/api/callbacks/early_stopping/" >}})
- [LearningRateScheduler]({{< relref "/docs/api/callbacks/learning_rate_scheduler/" >}})
- [ReduceLROnPlateau]({{< relref "/docs/api/callbacks/reduce_lr_on_plateau/" >}})
- [RemoteMonitor]({{< relref "/docs/api/callbacks/remote_monitor/" >}})
- [LambdaCallback]({{< relref "/docs/api/callbacks/lambda_callback/" >}})
- [TerminateOnNaN]({{< relref "/docs/api/callbacks/terminate_on_nan/" >}})
- [CSVLogger]({{< relref "/docs/api/callbacks/csv_logger/" >}})
- [ProgbarLogger]({{< relref "/docs/api/callbacks/progbar_logger/" >}})
- [SwapEMAWeights]({{< relref "/docs/api/callbacks/swap_ema_weights/" >}})

## Usage of callbacks via the built-in `fit()` loop

You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of a model:

```python
my_callbacks = [
    keras.callbacks.EarlyStopping(patience=2),
    keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
```

The relevant methods of the callbacks will then be called at each stage of the training.

## Using custom callbacks

Creating new callbacks is a simple and powerful way to customize a training loop.
Learn more about creating new callbacks in the guide
[Writing your own Callbacks]({{< relref "/docs/guides/writing_your_own_callbacks" >}}), and refer to
the documentation for [the base `Callback` class]({{< relref "/docs/api/callbacks/base_callback" >}}).
