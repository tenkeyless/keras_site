---
title: TensorBoard
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/callbacks/tensorboard.py#L17" >}}

### `TensorBoard` class

```python
keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
)
```

Enable visualizations for TensorBoard.

TensorBoard is a visualization tool provided with TensorFlow. A TensorFlow
installation is required to use this callback.

This callback logs events for TensorBoard, including:

- Metrics summary plots
- Training graph visualization
- Weight histograms
- Sampled profiling

When used in `model.evaluate()` or regular validation
in addition to epoch summaries, there will be a summary that records
evaluation metrics vs `model.optimizer.iterations` written. The metric names
will be prepended with `evaluation`, with `model.optimizer.iterations` being
the step in the visualized TensorBoard.

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:

```shell
tensorboard --logdir=path_to_your_logs
```

You can find more information about TensorBoard
[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

**Arguments**

- **log_dir**: the path of the directory where to save the log files to be
  parsed by TensorBoard. e.g.,
  `log_dir = os.path.join(working_dir, 'logs')`.
  This directory should not be reused by any other callbacks.
- **histogram_freq**: frequency (in epochs) at which to compute
  weight histograms for the layers of the model. If set to 0,
  histograms won't be computed. Validation data (or split) must be
  specified for histogram visualizations.
- **write_graph**: (Not supported at this time)
  Whether to visualize the graph in TensorBoard.
  Note that the log file can become quite large
  when `write_graph` is set to `True`.
- **write_images**: whether to write model weights to visualize as image in
  TensorBoard.
- **write_steps_per_second**: whether to log the training steps per second
  into TensorBoard. This supports both epoch and batch frequency
  logging.
- **update_freq**: `"batch"` or `"epoch"` or integer. When using `"epoch"`,
  writes the losses and metrics to TensorBoard after every epoch.
  If using an integer, let's say `1000`, all metrics and losses
  (including custom ones added by `Model.compile`) will be logged to
  TensorBoard every 1000 batches. `"batch"` is a synonym for 1,
  meaning that they will be written every batch.
  Note however that writing too frequently to TensorBoard can slow
  down your training, especially when used with distribution
  strategies as it will incur additional synchronization overhead.
  Batch-level summary writing is also available via `train_step`
  override. Please see
  [TensorBoard Scalars tutorial](https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging) # noqa: E501
  for more details.
- **profile_batch**: (Not supported at this time)
  Profile the batch(es) to sample compute characteristics.
  profile_batch must be a non-negative integer or a tuple of integers.
  A pair of positive integers signify a range of batches to profile.
  By default, profiling is disabled.
- **embeddings_freq**: frequency (in epochs) at which embedding layers will be
  visualized. If set to 0, embeddings won't be visualized.
- **embeddings_metadata**: Dictionary which maps embedding layer names to the
  filename of a file in which to save metadata for the embedding layer.
  In case the same metadata file is to be
  used for all embedding layers, a single filename can be passed.

**Examples**

```python
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
```

Custom batch-level summaries in a subclassed Model:

```python
class MyModel(keras.Model):
    def build(self, _):
        self.dense = keras.layers.Dense(10)
    def call(self, x):
        outputs = self.dense(x)
        tf.summary.histogram('outputs', outputs)
        return outputs
model = MyModel()
model.compile('sgd', 'mse')
tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)
model.fit(x_train, y_train, callbacks=[tb_callback])
```

Custom batch-level summaries in a Functional API Model:

```python
def my_summary(x):
    tf.summary.histogram('x', x)
    return x
inputs = keras.Input(10)
x = keras.layers.Dense(10)(inputs)
outputs = keras.layers.Lambda(my_summary)(x)
model = keras.Model(inputs, outputs)
model.compile('sgd', 'mse')
tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)
model.fit(x_train, y_train, callbacks=[tb_callback])
```

Profiling:

```python
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs', profile_batch=5)
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs', profile_batch=(10,20))
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
```
