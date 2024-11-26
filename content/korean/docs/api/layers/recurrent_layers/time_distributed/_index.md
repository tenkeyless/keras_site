---
title: TimeDistributed layer
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/rnn/time_distributed.py#L10" >}}

### `TimeDistributed` class

```python
keras.layers.TimeDistributed(layer, **kwargs)
```

This wrapper allows to apply a layer to every temporal slice of an input.

Every input should be at least 3D, and the dimension of index one of the first input will be considered to be the temporal dimension.

Consider a batch of 32 video samples, where each sample is a 128x128 RGB image with `channels_last` data format, across 10 timesteps. The batch input shape is `(32, 10, 128, 128, 3)`.

You can then use `TimeDistributed` to apply the same `Conv2D` layer to each of the 10 timesteps, independently:

```console
>>> inputs = layers.Input(shape=(10, 128, 128, 3), batch_size=32)
>>> conv_2d_layer = layers.Conv2D(64, (3, 3))
>>> outputs = layers.TimeDistributed(conv_2d_layer)(inputs)
>>> outputs.shape
(32, 10, 126, 126, 64)
```

Because `TimeDistributed` applies the same instance of `Conv2D` to each of the timestamps, the same set of weights are used at each timestamp.

**Arguments**

- **layer**: a [`keras.layers.Layer`]({{< relref "/docs/api/layers/base_layer#layer-class" >}}) instance.

**Call arguments**

- **inputs**: Input tensor of shape (batch, time, ...) or nested tensors, and each of which has shape (batch, time, ...).
- **training**: Python boolean indicating whether the layer should behave in training mode or in inference mode. This argument is passed to the wrapped layer (only if the layer supports this argument).
- **mask**: Binary tensor of shape `(samples, timesteps)` indicating whether a given timestep should be masked. This argument is passed to the wrapped layer (only if the layer supports this argument).
