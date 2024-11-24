---
title: Masking layer
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/core/masking.py#L7)

### `Masking` class

```python
keras.layers.Masking(mask_value=0.0, **kwargs)
```

Masks a sequence by using a mask value to skip timesteps.

For each timestep in the input tensor (dimension #1 in the tensor), if all values in the input tensor at that timestep are equal to `mask_value`, then the timestep will be masked (skipped) in all downstream layers (as long as they support masking).

If any downstream layer does not support masking yet receives such an input mask, an exception will be raised.

**Example**

Consider a NumPy data array `x` of shape `(samples, timesteps, features)`, to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you lack data for these timesteps. You can:

- Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
- Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

```python
samples, timesteps, features = 32, 10, 8
inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
inputs[:, 3, :] = 0.
inputs[:, 5, :] = 0.

model = keras.models.Sequential()
model.add(keras.layers.Masking(mask_value=0.)
model.add(keras.layers.LSTM(32))
output = model(inputs)
# The time step 3 and 5 will be skipped from LSTM calculation.
```

Note: in the Keras masking convention, a masked timestep is denoted by a mask value of `False`, while a non-masked (i.e. usable) timestep is denoted by a mask value of `True`.
