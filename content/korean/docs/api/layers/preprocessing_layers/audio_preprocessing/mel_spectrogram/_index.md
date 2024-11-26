---
title: MelSpectrogram layer
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/layers/preprocessing/mel_spectrogram.py#L9" >}}

### `MelSpectrogram` class

```python
keras.layers.MelSpectrogram(
    fft_length=2048,
    sequence_stride=512,
    sequence_length=None,
    window="hann",
    sampling_rate=16000,
    num_mel_bins=128,
    min_freq=20.0,
    max_freq=None,
    power_to_db=True,
    top_db=80.0,
    mag_exp=2.0,
    min_power=1e-10,
    ref_power=1.0,
    **kwargs
)
```

A preprocessing layer to convert raw audio signals to Mel spectrograms.

This layer takes `float32`/`float64` single or batched audio signal as inputs and computes the Mel spectrogram using Short-Time Fourier Transform and Mel scaling. The input should be a 1D (unbatched) or 2D (batched) tensor representing audio signals. The output will be a 2D or 3D tensor representing Mel spectrograms.

A spectrogram is an image-like representation that shows the frequency spectrum of a signal over time. It uses x-axis to represent time, y-axis to represent frequency, and each pixel to represent intensity. Mel spectrograms are a special type of spectrogram that use the mel scale, which approximates how humans perceive sound. They are commonly used in speech and music processing tasks like speech recognition, speaker identification, and music genre classification.

**References**

- [Spectrogram](https://en.wikipedia.org/wiki/Spectrogram),
- [Mel scale](https://en.wikipedia.org/wiki/Mel_scale).

**Examples**

**Unbatched audio signal**

```console
>>> layer = keras.layers.MelSpectrogram(num_mel_bins=64,
...                                     sampling_rate=8000,
...                                     sequence_stride=256,
...                                     fft_length=2048)
>>> layer(keras.random.uniform(shape=(16000,))).shape
(64, 63)
```

**Batched audio signal**

```console
>>> layer = keras.layers.MelSpectrogram(num_mel_bins=80,
...                                     sampling_rate=8000,
...                                     sequence_stride=128,
...                                     fft_length=2048)
>>> layer(keras.random.uniform(shape=(2, 16000))).shape
(2, 80, 125)
```

**Input shape**

1D (unbatched) or 2D (batched) tensor with shape:`(..., samples)`.

**Output shape**

2D (unbatched) or 3D (batched) tensor with shape:`(..., num_mel_bins, time)`.

**Arguments**

- **fft_length**: Integer, size of the FFT window.
- **sequence_stride**: Integer, number of samples between successive STFT columns.
- **sequence_length**: Integer, size of the window used for applying `window` to each audio frame. If `None`, defaults to `fft_length`.
- **window**: String, name of the window function to use. Available values are `"hann"` and `"hamming"`. If `window` is a tensor, it will be used directly as the window and its length must be `sequence_length`. If `window` is `None`, no windowing is used. Defaults to `"hann"`.
- **sampling_rate**: Integer, sample rate of the input signal.
- **num_mel_bins**: Integer, number of mel bins to generate.
- **min_freq**: Float, minimum frequency of the mel bins.
- **max_freq**: Float, maximum frequency of the mel bins. If `None`, defaults to `sampling_rate / 2`.
- **power_to_db**: If True, convert the power spectrogram to decibels.
- **top_db**: Float, minimum negative cut-off `max(10 * log10(S)) - top_db`.
- **mag_exp**: Float, exponent for the magnitude spectrogram. 1 for magnitude, 2 for power, etc. Default is 2.
- **ref_power**: Float, the power is scaled relative to it `10 * log10(S / ref_power)`.
- **min_power**: Float, minimum value for power and `ref_power`.
