---
title: AudioConverter layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/preprocessing/audio_converter.py#L12" >}}

### `AudioConverter` class

```python
keras_hub.layers.AudioConverter(**kwargs)
```

Convert raw audio for models that support audio input.

This class converts from raw audio tensors of any length, to preprocessed
audio for pretrained model inputs. It is meant to be a convenient way to
write custom preprocessing code that is not model specific. This layer
should be instantiated via the `from_preset()` constructor, which will
create the correct subclass of this layer for the model preset.

The layer will take as input a raw audio tensor with shape `(batch_size,
num_samples)`, and output a preprocessed audio input for modeling. The exact
structure of the preprocessed input will vary per model. Preprocessing
will often include computing a spectogram of the raw audio signal.

**Examples**

```python
# Load an audio converter from a preset.
converter = keras_hub.layers.AudioConverter.from_preset("whisper_base_en")
# Convert some raw audio input.
converter(np.ones(2, 1_000))
```
